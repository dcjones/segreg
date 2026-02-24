# Experimenting with writing this is pytorch instead to see how that would look.
from typing import cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from anndata import AnnData
from patsy import dmatrix
from patsy.design_info import DesignMatrix
from scipy.sparse import csr_matrix
from spatialdata import SpatialData
from torch.distributions import Beta, kl_divergence
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GCNConv


class Encoder(nn.Module):
    """
    Node-based GNN Encoder.
    Takes gene counts and design covariates, aggregates neighborhood,
    and outputs parameters for the latent normal distribution (mu, logstd).
    """

    def __init__(
        self, in_channels: int, hidden_channels: int = 64, latent_dim: int = 32
    ):
        super().__init__()
        self.lin_in = nn.Linear(in_channels, hidden_channels)
        self.conv1 = GCNConv(hidden_channels, hidden_channels)

        # Heads for variational parameters
        self.conv_mu = GCNConv(hidden_channels, latent_dim)
        self.conv_logstd = GCNConv(hidden_channels, latent_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.lin_in(x))
        x = F.relu(self.conv1(x, edge_index))
        mu = self.conv_mu(x, edge_index)
        logstd = self.conv_logstd(x, edge_index)
        return mu, logstd


class NodeDecoder(nn.Module):
    """
    Decodes the node latent representation back into true gene expression rates (lambda).
    """

    def __init__(self, latent_dim: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.lin1 = nn.Linear(latent_dim, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, z):
        h = F.relu(self.lin1(z))
        # softplus ensures predicted expression rates are > 0
        return F.softplus(self.lin2(h))


class EdgeDecoder(nn.Module):
    """
    Decodes pairs of latent node representations + prior into parameters
    for the posterior Beta distribution of diffusion coefficients (alpha).
    """

    def __init__(self, latent_dim: int, hidden_channels: int):
        super().__init__()
        # Input: z_i, z_j, prior_alpha
        self.lin1 = nn.Linear(latent_dim * 2 + 1, hidden_channels)
        self.lin_a = nn.Linear(hidden_channels, 1)
        self.lin_b = nn.Linear(hidden_channels, 1)

    def forward(self, z_src, z_dst, prior_alpha):
        h = torch.cat([z_src, z_dst, prior_alpha.unsqueeze(-1)], dim=-1)
        h = F.relu(self.lin1(h))

        # Beta distribution parameters a and b must be strictly positive
        a = F.softplus(self.lin_a(h)) + 1e-4
        b = F.softplus(self.lin_b(h)) + 1e-4
        return a, b


class SegregVAE(nn.Module):
    def __init__(
        self,
        n_genes: int,
        n_covariates: int,
        hidden_channels: int = 64,
        latent_dim: int = 32,
    ):
        super().__init__()
        in_channels = n_genes + n_covariates
        self.encoder = Encoder(in_channels, hidden_channels, latent_dim)
        self.node_decoder = NodeDecoder(latent_dim, hidden_channels, n_genes)
        self.edge_decoder = EdgeDecoder(latent_dim, hidden_channels)

    def reparameterize(self, mu, logstd):
        if self.training:
            std = torch.exp(logstd)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x, edge_index, prior_alpha):
        # 1. Encode into node latents
        mu, logstd = self.encoder(x, edge_index)
        z = self.reparameterize(mu, logstd)

        # 2. Decode true expression rates per cell
        lam = self.node_decoder(z)

        # 3. Decode edge diffusion parameters
        # edge_index is shape [2, E]. We assume edge_index[0] is source, edge_index[1] is target
        src, dst = edge_index
        z_src, z_dst = z[src], z[dst]
        a, b = self.edge_decoder(z_src, z_dst, prior_alpha)

        # Sample alpha during training, use mean during evaluation
        if self.training:
            alpha_dist = Beta(a, b)
            alpha = alpha_dist.rsample()
        else:
            alpha = a / (a + b)

        alpha = alpha.squeeze(-1)

        # 4. Forward Generative Model (Reconstruction)
        # x_hat_i = \lambda_i + \sum_j \alpha_{ij} \lambda_j
        messages = alpha.unsqueeze(-1) * lam[src]

        diffused = torch.zeros_like(lam)
        diffused.scatter_add_(0, dst.unsqueeze(-1).expand(-1, lam.size(1)), messages)

        x_hat = lam + diffused

        return x_hat, mu, logstd, a, b, alpha


class RegressionModel:
    X: csr_matrix
    data: Data
    design: DesignMatrix
    device: torch.device
    model: SegregVAE

    def __init__(
        self,
        data: SpatialData | AnnData,
        formula: str,
        batch_size: int | None = 4096,
    ):
        if isinstance(data, AnnData):
            adata = data
        elif isinstance(data, SpatialData):
            adata = data.tables["table"]
        else:
            raise ValueError("data must be an AnnData or SpatialData object")

        if "proseg_run" not in adata.uns:
            raise ValueError("This is not a proseg spatialdata file")

        m, n = adata.shape

        self.design = cast(DesignMatrix, dmatrix(formula, adata.obs))

        state_transitions = adata.obsp["state_transitions"]
        assert isinstance(state_transitions, csr_matrix)
        state_transitions_coo = state_transitions.tocoo()

        # NOTE: PyG NeighborLoader message passing treats edge_index[0] as source and edge_index[1] as target.
        # If 'state_transitions' maps transcript misassignment, transcripts flow from source to target.
        # We assign state_transitions_coo.col (original cell / source) to edge_index[0]
        # and state_transitions_coo.row (receiving cell / target) to edge_index[1].
        edge_index = np.stack(
            [state_transitions_coo.col, state_transitions_coo.row], axis=0
        )

        self.data = Data(
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            n_id=torch.arange(adata.n_obs),
            edge_attr=torch.tensor(state_transitions_coo.data, dtype=torch.float32),
            x=torch.tensor(np.asarray(self.design), dtype=torch.float32),
        )

        self.X = adata.X
        # cast to csr_matrix if we aren't already
        if not isinstance(self.X, csr_matrix):
            self.X = self.X.tocsr()

        # Initialize Model
        n_genes = self.X.shape[1]
        n_covariates = self.design.shape[1]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SegregVAE(n_genes, n_covariates).to(self.device)

    def fit(
        self,
        nepochs: int = 100,
        nneighbors: int = 10,
        batch_size: int = 1024,
        lr: float = 1e-3,
        kappa: float = 100.0,
    ):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        loader = NeighborLoader(
            self.data,
            num_neighbors=[nneighbors, nneighbors],
            batch_size=batch_size,
            input_nodes=None,
            shuffle=True,
        )

        self.model.train()

        for epoch in range(nepochs):
            total_loss = 0.0

            for batch in loader:
                optimizer.zero_grad()
                batch = batch.to(self.device)

                # Fetch sparse counts for the sampled nodes and convert to dense tensor for the model
                node_idx = batch.n_id.cpu().numpy()
                x_sub = self.X[node_idx].toarray()
                x_sub_tensor = torch.tensor(
                    x_sub, dtype=torch.float32, device=self.device
                )

                # Prepare Encoder Input: log1p(counts) concatenated with design matrix
                log_x = torch.log1p(x_sub_tensor)
                encoder_in = torch.cat([log_x, batch.x], dim=-1)

                prior_alpha = batch.edge_attr

                # Forward Pass
                x_hat, mu, logstd, a, b, alpha = self.model(
                    encoder_in, batch.edge_index, prior_alpha
                )

                # Masking: We only calculate loss for the "target" nodes in the center of the sampled subgraph.
                # PyG places these first in the batch up to `batch.batch_size`
                target_mask = torch.arange(batch.batch_size, device=self.device)
                x_target = x_sub_tensor[target_mask]
                x_hat_target = x_hat[target_mask]

                # 1. Reconstruction Loss (Negative Log-Likelihood of Poisson)
                # Poisson log PMF: x * log(lambda) - lambda - log(x!)
                # Minimizing NLL: -x * log(lambda) + lambda
                loss_recon = (
                    (x_hat_target - x_target * torch.log(x_hat_target + 1e-8))
                    .sum(dim=-1)
                    .mean()
                )

                # 2. Node KL Divergence (Standard Normal Prior)
                mu_target = mu[target_mask]
                logstd_target = logstd[target_mask]
                kl_z = (
                    -0.5
                    * torch.sum(
                        1
                        + 2 * logstd_target
                        - mu_target.pow(2)
                        - torch.exp(2 * logstd_target),
                        dim=1,
                    ).mean()
                )

                # 3. Edge KL Divergence (Beta Prior from Proseg)
                dst = batch.edge_index[1]
                edge_mask = dst < batch.batch_size

                if edge_mask.sum() > 0:
                    a_target = a[edge_mask]
                    b_target = b[edge_mask]
                    prior_alpha_target = prior_alpha[edge_mask].unsqueeze(-1)

                    # Clamp prior to avoid 0 or 1 edge cases for the Beta distribution
                    prior_alpha_target = torch.clamp(
                        prior_alpha_target, 1e-4, 1.0 - 1e-4
                    )

                    # Prior distribution parameterized using kappa (concentration) and prior_alpha (mean)
                    prior_a = kappa * prior_alpha_target
                    prior_b = kappa * (1.0 - prior_alpha_target)

                    q_alpha = Beta(a_target, b_target)
                    p_alpha = Beta(prior_a, prior_b)

                    # Normalize by batch size to keep loss scale invariant
                    kl_alpha = kl_divergence(q_alpha, p_alpha).sum() / batch.batch_size
                else:
                    kl_alpha = torch.tensor(0.0, device=self.device)

                loss = loss_recon + kl_z + kl_alpha
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch} | Loss: {total_loss / len(loader):.4f}")
