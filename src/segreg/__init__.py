# Experimenting with writing this is pytorch instead to see how that would look.
from typing import cast

import numpy as np
import pandas as pd
import scipy.stats as stats
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
    Decodes the node latent representation back into unconstrained expression rates (rho).
    """

    def __init__(self, latent_dim: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.lin1 = nn.Linear(latent_dim, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels, bias=False)

    def forward(self, z):
        h = F.relu(self.lin1(z))
        # Return unconstrained values to be added to the regression term
        return self.lin2(h)


class EdgeDecoder(nn.Module):
    """
    Decodes pairs of latent node representations + prior into parameters
    for the posterior Beta distribution of diffusion coefficients (alpha).
    """

    def __init__(self, latent_dim: int, hidden_channels: int, n_genes: int):
        super().__init__()
        # Input: z_i, z_j, prior_alpha
        self.lin1 = nn.Linear(latent_dim * 2 + n_genes, hidden_channels)
        self.lin_a = nn.Linear(hidden_channels, 1)
        self.lin_b = nn.Linear(hidden_channels, 1)

    def forward(self, z_src, z_dst, prior_alpha):
        h = torch.cat([z_src, z_dst, prior_alpha], dim=-1)
        h = F.relu(self.lin1(h))

        # Beta distribution parameters a and b must be strictly positive
        # We add 1.0 to ensure they never create an asymptote at 0 or 1
        a = F.softplus(self.lin_a(h)) + 1.0
        b = F.softplus(self.lin_b(h)) + 1.0
        return a, b


class SegregVAE(nn.Module):
    def __init__(
        self,
        n_genes: int,
        n_covariates: int,
        hidden_channels: int = 64,
        latent_dim: int = 32,
        kappa: float = 100.0,
        include_diffusion: bool = True,
        log_mean_expr: torch.Tensor | None = None,
    ):
        super().__init__()
        self.include_diffusion = include_diffusion
        in_channels = n_genes + n_covariates
        self.encoder = Encoder(in_channels, hidden_channels, latent_dim)
        self.node_decoder = NodeDecoder(latent_dim, hidden_channels, n_genes)

        if self.include_diffusion:
            self.edge_decoder = EdgeDecoder(latent_dim, hidden_channels, n_genes)

            # Global concentration parameter for Beta prior (kappa).
            # We model it as a point estimate (MLE) to be inferred during training.
            # Use softplus to ensure it stays positive.
            self.kappa_unconstrained = nn.Parameter(torch.tensor(kappa))

        print(log_mean_expr)

        # Empirical gene baseline to help the model start at the correct scale
        if log_mean_expr is not None:
            self.gene_bias = nn.Parameter(log_mean_expr.clone())
        else:
            self.gene_bias = nn.Parameter(torch.zeros(n_genes))

        # Global regression parameters (Surrogate model for Variational Inference)
        self.beta_mu = nn.Parameter(torch.zeros(n_covariates, n_genes))
        # Initialize logstd to a small value so that initial samples are close to the mean
        self.beta_logstd = nn.Parameter(torch.full((n_covariates, n_genes), -3.0))

    @property
    def kappa(self):
        if not self.include_diffusion:
            return None
        return F.softplus(self.kappa_unconstrained)

    def reparameterize(self, mu, logstd):
        if self.training:
            std = torch.exp(logstd)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x, covariates, edge_index, prior_alpha=None):
        # 1. Encode into node latents
        mu, logstd = self.encoder(x, edge_index)
        z = self.reparameterize(mu, logstd)

        # 2. Decode unconstrained expression rates per cell
        rho = self.node_decoder(z)

        # 2.5 Sample global regression coefficients
        if self.training:
            beta_std = torch.exp(self.beta_logstd)
            eps = torch.randn_like(beta_std)
            beta = self.beta_mu + eps * beta_std
        else:
            beta = self.beta_mu

        # Compute predicted expression rates (lambda)
        # Use clamp to prevent overflow when applying exp
        lam = torch.exp(
            torch.clamp(rho + covariates @ beta + self.gene_bias, min=-15.0, max=15.0)
        )

        if self.include_diffusion and prior_alpha is not None:
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

            # Enforce physical conservation of mass: a cell cannot diffuse more than 100% of its transcripts.
            # Sum the inferred alpha values over all outgoing edges from each source node.
            total_alpha = torch.zeros(lam.size(0), device=lam.device)
            total_alpha.scatter_add_(0, src, alpha)
            
            # If total_alpha > 1.0, we normalize the outgoing alphas down. 
            # If < 1.0, we leave them (allowing loss to background).
            normalization = torch.clamp(total_alpha, min=1.0)
            alpha_normalized = alpha / normalization[src]

            # 4. Forward Generative Model (Reconstruction)
            # The edge_index and alpha values include self-loops (i->i).
            # Therefore, the total transcripts ending up in cell i is just the sum of messages.
            # x_hat_i = \sum_{j} \alpha_{ji} \lambda_j
            messages = alpha_normalized.unsqueeze(-1) * lam[src]

            diffused = torch.zeros_like(lam)
            diffused.scatter_add_(0, dst.unsqueeze(-1).expand(-1, lam.size(1)), messages)

            x_hat = diffused
        else:
            x_hat = lam
            a = b = alpha = None

        return x_hat, mu, logstd, a, b, alpha


class RegressionModel:
    X: csr_matrix
    data: Data
    design: DesignMatrix
    device: torch.device
    model: SegregVAE
    state_transitions_t: csr_matrix
    m: int
    n: int

    def __init__(
        self,
        data: SpatialData | AnnData,
        formula: str,
        batch_size: int | None = 4096,
        include_diffusion: bool = True,
        hidden_channels: int = 64,
        latent_dim: int = 32,
        kappa: float = 100.0,
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

        self.var_names = adata.var_names
        self.obs_names = adata.obs_names

        self.design = cast(DesignMatrix, dmatrix(formula, adata.obs))

        # This is a flattened 3d array giving per-gene state transition probabilities
        state_transitions = adata.varm["state_transitions"]
        assert state_transitions.shape == (n, m * m)
        assert isinstance(state_transitions, csr_matrix)

        # define a graph where there is an edge if there were any transcript transitions between cells
        unique_indices = np.unique(state_transitions.indices, sorted=True)
        col = unique_indices % m
        row = unique_indices // m
        edge_index = np.stack([col, row], axis=0)

        self.data = Data(
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            n_id=torch.arange(adata.n_obs),
            # edge_attr=torch.tensor(state_transitions_coo.data, dtype=torch.float32),
            x=torch.tensor(np.asarray(self.design), dtype=torch.float32),
        )

        self.state_transitions_t = state_transitions.transpose().tocsr()

        self.data = Data(
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            n_id=torch.arange(adata.n_obs),
            x=torch.tensor(np.asarray(self.design), dtype=torch.float32),
        )

        self.X = adata.X
        # cast to csr_matrix if we aren't already
        if not isinstance(self.X, csr_matrix):
            self.X = self.X.tocsr()

        # Compute empirical log mean expression for initialization
        # Add a small epsilon to avoid log(0)
        mean_expr = torch.tensor(
            np.asarray(self.X.mean(axis=0)).squeeze(), dtype=torch.float32
        )
        log_mean_expr = torch.log(mean_expr + 1e-4)

        # Initialize Model
        n_genes = self.X.shape[1]
        n_covariates = self.design.shape[1]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SegregVAE(
            n_genes,
            n_covariates,
            hidden_channels=hidden_channels,
            latent_dim=latent_dim,
            kappa=kappa,
            include_diffusion=include_diffusion,
            log_mean_expr=log_mean_expr,
        ).to(self.device)
        self.m = m
        self.n = n

    def fit(
        self,
        nepochs: int = 100,
        nneighbors: int = 10,
        batch_size: int = 1024,
        lr: float = 1e-3,
        beta_kl: float = 0.01,
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

                if self.model.include_diffusion:
                    # Fetch edge transition weights
                    global_src = batch.n_id[batch.edge_index[0, :]]
                    global_dst = batch.n_id[batch.edge_index[1, :]]
                    encoded_edge_index = (
                        (global_src + global_dst * self.m).cpu().numpy()
                    )

                    prior_alpha = torch.tensor(
                        self.state_transitions_t[encoded_edge_index, :].todense(),
                        dtype=torch.float32,
                        device=self.device,
                    )
                else:
                    prior_alpha = None

                # Prepare Encoder Input: log1p(counts) concatenated with design matrix
                log_x = torch.log1p(x_sub_tensor)
                encoder_in = torch.cat([log_x, batch.x], dim=-1)

                # Forward Pass
                x_hat, mu, logstd, a, b, alpha = self.model(
                    encoder_in, batch.x, batch.edge_index, prior_alpha
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
                if self.model.include_diffusion:
                    dst = batch.edge_index[1]
                    edge_mask = dst < batch.batch_size

                    if edge_mask.sum() > 0:
                        a_target = a[edge_mask]
                        b_target = b[edge_mask]
                        prior_alpha_target = prior_alpha[edge_mask]

                        # Clamp prior to avoid 0 or 1 edge cases for the Beta distribution
                        prior_alpha_target = torch.clamp(
                            prior_alpha_target, 1e-4, 1.0 - 1e-4
                        )

                        # Use the inferred kappa concentration parameter
                        kappa = self.model.kappa

                        # Prior distribution parameterized using kappa (concentration) and prior_alpha (mean)
                        # We add 1.0 to ensure the prior never has an asymptote at 0 or 1
                        prior_a = 1.0 + kappa * prior_alpha_target
                        prior_b = 1.0 + kappa * (1.0 - prior_alpha_target)

                        q_alpha = Beta(a_target, b_target)
                        p_alpha = Beta(prior_a, prior_b)
                        # Normalize by batch size to keep loss scale invariant
                        kl_alpha = (
                            kl_divergence(q_alpha, p_alpha).sum() / batch.batch_size
                        )
                    else:
                        kl_alpha = torch.tensor(0.0, device=self.device)
                else:
                    kl_alpha = torch.tensor(0.0, device=self.device)

                # 4. Global Regression Parameters KL Divergence (Standard Normal Prior)
                kl_beta = (
                    -0.5
                    * torch.sum(
                        1
                        + 2 * self.model.beta_logstd
                        - self.model.beta_mu.pow(2)
                        - torch.exp(2 * self.model.beta_logstd)
                    )
                ) / self.m

                loss = loss_recon + beta_kl * (kl_z + kl_alpha + kl_beta)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if self.model.include_diffusion:
                kappa_val = self.model.kappa.item()
                kappa_str = f", Kappa: {kappa_val:.2f}"
            else:
                kappa_str = ""

            print(
                f"Epoch {epoch} | Loss: {total_loss / len(loader):.4f} "
                f"(Recon: {loss_recon.item():.4f}, KL_z: {kl_z.item():.4f}, "
                f"KL_alpha: {kl_alpha.item():.4f}, KL_beta: {kl_beta.item():.4f}{kappa_str})"
            )

    def get_regression_coefficients(
        self, credible_interval: float | None = None
    ) -> pd.DataFrame:
        """
        Returns a pandas DataFrame containing the posterior mean point estimates for the
        regression coefficients. If a credible_interval is provided (e.g. 0.95), it also
        calculates and includes the corresponding lower and upper bounds.
        """
        beta_mu = self.model.beta_mu.detach().cpu().numpy()
        covariate_names = self.design.design_info.column_names
        genes = self.var_names

        df = (
            pd.DataFrame(beta_mu, index=covariate_names, columns=genes)
            .melt(ignore_index=False, var_name="Gene", value_name="Mean")
            .reset_index(names="Covariate")
        )

        if credible_interval is not None:
            beta_std = torch.exp(self.model.beta_logstd).detach().cpu().numpy()
            alpha = 1.0 - credible_interval
            z = stats.norm.ppf(1.0 - alpha / 2.0)

            lower = (
                pd.DataFrame(
                    beta_mu - z * beta_std, index=covariate_names, columns=genes
                )
                .melt(ignore_index=False, var_name="Gene", value_name="Lower")
                .reset_index(names="Covariate")
            )

            upper = (
                pd.DataFrame(
                    beta_mu + z * beta_std, index=covariate_names, columns=genes
                )
                .melt(ignore_index=False, var_name="Gene", value_name="Upper")
                .reset_index(names="Covariate")
            )

            df["Lower"] = lower["Lower"]
            df["Upper"] = upper["Upper"]

        return df

    def get_corrected_expression(
        self,
        threshold: float = 1e-4,
        batch_size: int = 4096,
        nneighbors: int = 10,
        n_samples: int = 10,
    ) -> csr_matrix:
        """
        Returns the 'corrected' estimates of gene expression rates (lambda), which
        represent the modeled expression prior to diffusion effects from neighboring cells.
        The result is a sparse CSR matrix with values below `threshold` set to 0.
        By default, it uses Monte Carlo sampling (`n_samples`) to estimate the expected rates.
        """
        self.model.eval()

        loader = NeighborLoader(
            self.data,
            num_neighbors=[nneighbors, nneighbors],
            batch_size=batch_size,
            input_nodes=None,
            shuffle=False,
        )

        rows = []
        cols = []
        data = []

        current_row = 0
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)

                # Fetch sparse counts for the sampled nodes and convert to dense tensor
                node_idx = batch.n_id.cpu().numpy()
                x_sub = self.X[node_idx].toarray()
                x_sub_tensor = torch.tensor(
                    x_sub, dtype=torch.float32, device=self.device
                )

                log_x = torch.log1p(x_sub_tensor)
                encoder_in = torch.cat([log_x, batch.x], dim=-1)

                # Forward pass through encoder
                mu, logstd = self.model.encoder(encoder_in, batch.edge_index)

                # PyG places the target nodes first in the batch up to `batch.batch_size`
                target_mask = slice(0, batch.batch_size)
                mu_target = mu[target_mask]
                logstd_target = logstd[target_mask]
                covariates_target = batch.x[target_mask]
                beta = self.model.beta_mu

                std_target = torch.exp(logstd_target)

                lam_target_sum = torch.zeros(
                    (batch.batch_size, self.n), device=self.device
                )

                # Monte Carlo sampling to compute expected rates
                for _ in range(n_samples):
                    z_target = mu_target + torch.randn_like(std_target) * std_target
                    rho_target = self.model.node_decoder(z_target)

                    lam_target_sum += torch.exp(
                        torch.clamp(
                            rho_target
                            + covariates_target @ beta
                            + self.model.gene_bias,
                            min=-15.0,
                            max=15.0,
                        )
                    )

                lam_target = lam_target_sum / n_samples
                lam_np = lam_target.cpu().numpy()

                # Thresholding
                lam_np[lam_np < threshold] = 0.0

                # Extract non-zero elements to build CSR matrix
                r, c = np.nonzero(lam_np)
                v = lam_np[r, c]

                rows.append(r + current_row)
                cols.append(c)
                data.append(v)

                current_row += batch.batch_size

        if rows:
            rows = np.concatenate(rows)
            cols = np.concatenate(cols)
            data = np.concatenate(data)
        else:
            rows = np.array([], dtype=int)
            cols = np.array([], dtype=int)
            data = np.array([], dtype=float)

        return csr_matrix((data, (rows, cols)), shape=(self.m, self.n))
