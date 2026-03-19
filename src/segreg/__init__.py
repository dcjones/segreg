# Experimenting with writing this is pytorch instead to see how that would look.
import math
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
from tqdm import tqdm


class Encoder(nn.Module):
    """
    Node-based GNN Encoder with skip connections and LayerNorm.
    Takes gene counts and design covariates, aggregates neighborhood,
    and outputs parameters for the latent normal distribution (mu, logstd).
    """

    def __init__(
        self, in_channels: int, hidden_channels: int = 128, latent_dim: int = 64
    ):
        super().__init__()
        # Initial transformation to latent-like space, using 2 layers for more capacity
        self.lin_in = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
        )
        # GCN to capture neighborhood context
        self.conv = GCNConv(hidden_channels, hidden_channels)

        # Separate heads for mu and logstd, using both local and spatial features
        # (Concatenation provides a clear path for the model to preserve cell-specific features)
        self.mu_head = nn.Linear(hidden_channels * 2, latent_dim)
        self.logstd_head = nn.Linear(hidden_channels * 2, latent_dim)

    def forward(self, x, edge_index):
        h = self.lin_in(x)
        h_spatial = F.relu(self.conv(h, edge_index))

        # Combine local and spatial features
        h_combined = torch.cat([h, h_spatial], dim=-1)

        mu = self.mu_head(h_combined)
        logstd = self.logstd_head(h_combined)
        return mu, logstd


class NodeDecoder(nn.Module):
    """
    Decodes the node latent representation back into unconstrained expression rates (rho).
    Increased capacity with LayerNorm and more layers.
    """

    def __init__(self, latent_dim: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels, bias=False),
        )

    def forward(self, z):
        # Return unconstrained values to be added to the regression term
        return self.net(z)


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
        include_size_factor: bool = True,
        log_mean_expr: torch.Tensor | None = None,
        beta_init: torch.Tensor | None = None,
        beta_logstd_init: torch.Tensor | None = None,
    ):
        super().__init__()
        self.include_diffusion = include_diffusion
        self.include_size_factor = include_size_factor
        # When using a size factor the encoder receives only expression data (not covariates),
        # so the latent variable cannot absorb systematic covariate effects (e.g. tumor adjacency)
        # and the regression coefficients remain the sole owner of those effects.
        in_channels = n_genes if include_size_factor else n_genes + n_covariates
        self.encoder = Encoder(in_channels, hidden_channels, latent_dim)
        self.node_decoder = NodeDecoder(latent_dim, hidden_channels, n_genes)

        if self.include_diffusion:
            self.edge_decoder = EdgeDecoder(latent_dim, hidden_channels, n_genes)

            # Global concentration parameter for Beta prior (kappa).
            # We model it as a point estimate (MLE) to be inferred during training.
            # Use softplus to ensure it stays positive.
            self.kappa_unconstrained = nn.Parameter(torch.tensor(kappa))

        # Empirical gene baseline to help the model start at the correct scale
        if log_mean_expr is not None:
            self.gene_bias = nn.Parameter(log_mean_expr.clone())
        else:
            self.gene_bias = nn.Parameter(torch.zeros(n_genes))

        # Per-gene negative binomial dispersion: r = softplus(log_r).
        # NB variance = mu + mu^2/r; as r -> inf this reduces to Poisson.
        # Initialize to softplus_inv(10) ≈ 9.3 so training starts near-Poisson
        # and learns overdispersion where the data supports it.
        self.log_r = nn.Parameter(torch.full((n_genes,), 9.3))

        # Global regression parameters (Surrogate model for Variational Inference)
        if beta_init is not None:
            self.beta_mu = nn.Parameter(beta_init.clone())
        else:
            self.beta_mu = nn.Parameter(torch.zeros(n_covariates, n_genes))
        # Initialize logstd from Fisher-info-based posterior variance when available;
        # otherwise fall back to a small fixed value.
        if beta_logstd_init is not None:
            self.beta_logstd = nn.Parameter(beta_logstd_init.clone())
        else:
            self.beta_logstd = nn.Parameter(torch.full((n_covariates, n_genes), -3.0))

    @property
    def kappa(self):
        if not self.include_diffusion:
            return None
        return F.softplus(self.kappa_unconstrained)

    def reparameterize(self, mu, logstd):
        if self.training:
            # Clamp to prevent bfloat16 overflow: large mu/logstd → Inf → mixed-sign
            # Inf in downstream linear layers → Inf + (-Inf) = NaN.
            std = torch.exp(logstd.clamp(max=10.0))
            eps = torch.randn_like(std)
            return (mu + eps * std).clamp(-50.0, 50.0)
        return mu.clamp(-50.0, 50.0)

    def forward(
        self, x, covariates, edge_index, prior_alpha=None, log_size_factor=None
    ):
        # 1. Encode into node latents
        mu, logstd = self.encoder(x, edge_index)
        z = self.reparameterize(mu, logstd)

        # 2. Decode unconstrained expression rates per cell
        rho = self.node_decoder(z)

        # 2.5 Sample global regression coefficients
        if self.training:
            # Clamp logstd before exp to prevent bfloat16 overflow → Inf in beta_std
            beta_std = torch.exp(self.beta_logstd.clamp(max=4.0))
            eps = torch.randn_like(beta_std)
            beta = self.beta_mu + eps * beta_std
        else:
            beta = self.beta_mu

        # Compute predicted expression rates (lambda).
        # When include_size_factor is True, log_size_factor is added as a fixed offset
        # so that beta captures rates per unit of total expression, not raw counts.
        log_rate = rho + covariates @ beta + self.gene_bias
        if self.include_size_factor and log_size_factor is not None:
            log_rate = log_rate + log_size_factor.unsqueeze(-1)
        lam = torch.exp(torch.clamp(log_rate, min=-15.0, max=15.0))

        if self.include_diffusion and prior_alpha is not None:
            # 3. Decode edge diffusion parameters
            # edge_index is shape [2, E]. We assume edge_index[0] is source, edge_index[1] is target
            src, dst = edge_index
            z_src, z_dst = z[src], z[dst]
            a, b = self.edge_decoder(z_src, z_dst, prior_alpha)

            # Sample alpha during training, use mean during evaluation.
            # Cast to float32 before sampling: Beta.rsample() via Gamma can produce
            # NaN in bfloat16 for some parameter combinations.
            if self.training:
                if a.numel() > 0:
                    # Clamp to avoid numerically unstable Gamma sampler for very
                    # large concentrations (can produce Inf → NaN via Inf/Inf).
                    a_f = a.float().clamp(min=1e-3, max=1e4)
                    b_f = b.float().clamp(min=1e-3, max=1e4)
                    alpha_dist = Beta(a_f, b_f)
                    alpha = alpha_dist.rsample().to(a.dtype)
                else:
                    alpha = torch.empty_like(a)
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
            diffused.scatter_add_(
                0, dst.unsqueeze(-1).expand(-1, lam.size(1)), messages
            )

            x_hat = diffused
        else:
            x_hat = lam
            a = b = alpha = None

        return x_hat, mu, logstd, a, b, alpha, beta


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
        include_size_factor: bool = True,
        sf_sigma: float = 0.5,
        hidden_channels: int = 128,
        latent_dim: int = 64,
        kappa: float = 10.0,
        beta_prior_scale: float = 1.0,
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

        design_df = dmatrix(formula, adata.obs, return_type="dataframe")
        self.design = cast(DesignMatrix, design_df)

        # dmatrix drops rows with NaN by default; expand back to m rows with zeros
        # so that Data.x aligns with n_id (which covers all m cells).
        if design_df.shape[0] < m:
            design_df = design_df.reindex(adata.obs.index, fill_value=0.0)

        # This is a flattened 3d array giving per-gene state transition probabilities
        state_transitions = adata.varm["state_transitions"]

        # SciPy/AnnData zarr loaders sometimes use int32 for indices, which overflows if m*m > 2.14 billion.
        # CSR row indices must be monotonically increasing, so we can detect and fix negative wraps.
        if state_transitions.indices.dtype == np.int32 and m * m > 2147483647:
            print("HERE")
            indices_64 = state_transitions.indices.astype(np.int64)
            for i in range(len(state_transitions.indptr) - 1):
                start, end = (
                    state_transitions.indptr[i],
                    state_transitions.indptr[i + 1],
                )
                if start == end:
                    continue
                row_indices = indices_64[start:end]
                diffs = np.diff(row_indices)
                wraps = (diffs < 0).astype(np.int64)
                if wraps.any():
                    wrap_counts = np.cumsum(wraps)
                    row_indices[1:] += wrap_counts * (2**32)
                    indices_64[start:end] = row_indices

            state_transitions = csr_matrix(
                (state_transitions.data, indices_64, state_transitions.indptr),
                shape=(n, m * m),
            )

        assert state_transitions.shape == (n, m * m)
        assert isinstance(state_transitions, csr_matrix)

        # define a graph where there is an edge if there were any transcript transitions between cells
        self.unique_edge_indices = np.unique(state_transitions.indices)
        col = self.unique_edge_indices % m
        row = self.unique_edge_indices // m

        edge_index = np.stack([col, row], axis=0)

        # Compress state transitions to only include columns with any non-zero entries
        new_indices = np.searchsorted(
            self.unique_edge_indices, state_transitions.indices
        )
        compressed_state_transitions = csr_matrix(
            (state_transitions.data, new_indices, state_transitions.indptr),
            shape=(n, len(self.unique_edge_indices)),
        )
        self.state_transitions_t = compressed_state_transitions.transpose().tocsr()

        self.X = adata.X
        # cast to csr_matrix if we aren't already
        if not isinstance(self.X, csr_matrix):
            self.X = self.X.tocsr()

        # Per-cell log size factor: use cell volume from proseg if available,
        # otherwise fall back to total transcript count. Volume is the geometrically
        # correct size measurement; counts/volume is flat across cell groups.
        if include_size_factor and "volume" in adata.obs.columns:
            cell_size = np.asarray(adata.obs["volume"]).squeeze().astype(np.float64)
        else:
            cell_size = np.asarray(self.X.sum(axis=1)).squeeze().astype(np.float64)
        log_size_factors = np.log(cell_size + 1e-8).astype(np.float32)

        self.data = Data(
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            n_id=torch.arange(adata.n_obs),
            x=torch.tensor(np.asarray(design_df), dtype=torch.float32),
            # Prior for the learned size factor: fixed geometric measurement from proseg.
            log_sf_prior=torch.tensor(log_size_factors, dtype=torch.float32),
        )

        # Learnable per-cell log size factor, initialized to the volume-based prior.
        # During training it is regularized toward the prior with a Gaussian penalty,
        # allowing small per-cell adjustments while staying anchored to cell geometry.
        self.sf_sigma = sf_sigma
        self.include_size_factor = include_size_factor
        if include_size_factor:
            self.log_sf_embed = nn.Embedding(m, 1)
            self.log_sf_embed.weight.data = torch.tensor(
                log_size_factors, dtype=torch.float32
            ).unsqueeze(-1)

        # Compute empirical log mean expression for initialization.
        # With a size factor, gene_bias should reflect the mean normalized rate:
        # E[counts_g] = mean_sf * exp(gene_bias_g), so gene_bias = log(mean_expr/mean_sf).
        mean_expr = torch.tensor(
            np.asarray(self.X.mean(axis=0)).squeeze(), dtype=torch.float32
        )
        if include_size_factor:
            mean_sf = float(cell_size.mean())
            log_mean_expr = torch.log(mean_expr / mean_sf + 1e-8)
        else:
            log_mean_expr = torch.log(mean_expr + 1e-4)

        # OLS initialization for beta_mu: solve log(x/sf + pseudocount) ≈ gene_bias + design @ beta
        # One QR factorization of the (m × n_cov) design matrix solves all genes at once.
        # This gives beta a near-correct starting point so training converges much faster.
        n_genes = self.X.shape[1]
        n_covariates = self.design.shape[1]
        print("Computing OLS initialization for beta...")
        X_dense = self.X.toarray().astype(np.float32)  # (m, n_genes)
        sf_col = np.exp(log_size_factors).reshape(-1, 1)  # (m, 1)
        # Use a per-gene pseudocount of 0.5 * mean_rate in rate space.
        # This ensures zero-count cells always get y_resid ≈ log(0.5) ≈ -0.7 regardless
        # of the gene's mean expression level, avoiding the extreme negative residuals
        # that a tiny fixed pseudocount (1e-8) causes for lowly-expressed genes (which
        # drove the OLS to initialize with large negative betas / spurious downregulation).
        # Being in rate space (not count space) also means the pseudocount is independent
        # of cell size, preventing the sf-correlated bias that additive count pseudocounts
        # introduce when cell size correlates with covariates.
        mean_rate = (  # (1, n_genes) mean expression rate across cells
            np.asarray(self.X.mean(axis=0)).squeeze().astype(np.float32).reshape(1, -1)
            / float(sf_col.mean())
        )
        gene_pseudocount = 0.5 * np.maximum(mean_rate, 2e-8)  # (1, n_genes)
        y_log = np.log(X_dense / sf_col + gene_pseudocount)  # (m, n_genes)
        y_resid = y_log - log_mean_expr.numpy()  # subtract gene_bias
        design_np = np.asarray(design_df, dtype=np.float64)  # (m, n_cov)
        beta_init_np, _, _, _ = np.linalg.lstsq(
            design_np, y_resid.astype(np.float64), rcond=None
        )
        beta_init_np = beta_init_np.astype(np.float32)
        print("Done.")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Cauchy prior scale for regression coefficients.
        # The intercept gets a wide scale (10.0) so it is essentially unconstrained;
        # all other covariates get beta_prior_scale (γ) to encourage shrinkage toward zero
        # while the heavy Cauchy tails still allow genuinely large fold changes.
        #
        # Gene-specific scaling: gamma_g ∝ sqrt(mean_rate_g / median_rate).
        # Motivation: the NB reconstruction gradient w.r.t. beta[k, g] scales with
        # mean_rate_g, so any systematic effect (real biology or sf artifact) survives
        # Cauchy shrinkage preferentially for highly expressed genes, producing a spurious
        # positive correlation between baseline expression and estimated fold changes.
        # Scaling gamma by sqrt(mean_rate_g / median_rate) matches the typical statistical
        # uncertainty in estimating a log fold change (∝ 1/sqrt(n * mean_rate_g)), so the
        # effective prior-to-signal ratio is equalized across the expression range.
        covariate_names = list(self.design.design_info.column_names)
        scale_per_cov = np.array(
            [
                10.0 if name == "Intercept" else beta_prior_scale
                for name in covariate_names
            ],
            dtype=np.float32,
        )
        mean_rate_1d = mean_rate.squeeze()  # (n_genes,)
        expressed_rates = mean_rate_1d[mean_rate_1d > 1e-6]
        median_rate = float(np.median(expressed_rates)) if len(expressed_rates) > 0 else 1e-4
        gene_scale = np.sqrt(
            np.maximum(mean_rate_1d, 1e-8) / median_rate
        ).clip(0.1, 10.0).astype(np.float32)  # (n_genes,); clipped to [0.1, 10]
        # (n_cov, n_genes): per-covariate base scale * per-gene expression scale
        beta_prior_scale_matrix = scale_per_cov[:, None] * gene_scale[None, :]
        self.beta_prior_scale_t = torch.tensor(
            beta_prior_scale_matrix, dtype=torch.float32
        ).to(self.device)

        # Fisher-information-based initialization for beta_mu and beta_logstd.
        #
        # For each (covariate k, gene g), the Fisher information of the NB
        # likelihood w.r.t. beta[k,g] is approximately:
        #   I[k,g] = m * mean(design[:,k]^2) * r_init * mean_rate_g / (r_init + mean_rate_g)
        #
        # This drives two calibrated initializations:
        #
        # 1. Shrink beta_mu toward 0 (posterior mean under Gaussian-prior approximation):
        #      beta_init[k,g] = beta_ols[k,g] * gamma[k,g]^2 / (gamma[k,g]^2 + 1/I[k,g])
        #    For near-zero genes (I≈0), the full OLS estimate is noise and gets shrunk
        #    to ~0. For well-expressed genes (I large), shrinkage is negligible.
        #
        # 2. Initialize beta_logstd to the posterior standard deviation:
        #      sigma_post[k,g] = sqrt(gamma[k,g]^2 / (1 + gamma[k,g]^2 * I[k,g]))
        #    For near-zero genes: sigma_post ≈ gamma (falls back to prior width).
        #    For well-expressed genes: sigma_post ≈ 1/sqrt(I) (data-driven precision).
        #
        # Because I scales with m and mean_rate_g this self-calibrates across
        # datasets and expression levels without any arbitrary tuning constants.
        r_init = float(np.log1p(np.exp(9.3)))  # initial NB dispersion ≈ 10
        fisher_per_cell_g = (
            r_init * mean_rate_1d / (r_init + mean_rate_1d + 1e-10)
        )  # (n_genes,)
        design_cov_var = np.mean(design_np**2, axis=0).astype(np.float32)  # (n_cov,)
        fisher_info = np.outer(
            m * design_cov_var, fisher_per_cell_g
        )  # (n_cov, n_genes)
        gamma_sq = beta_prior_scale_matrix**2  # (n_cov, n_genes)
        sigma_ols_sq = 1.0 / np.maximum(fisher_info, 1e-6)  # (n_cov, n_genes)
        # Posterior mean: shrink OLS toward 0 proportional to data uncertainty
        shrink = gamma_sq / (gamma_sq + sigma_ols_sq)  # in [0, 1]
        beta_init_np = beta_init_np * shrink
        # Posterior variance: harmonic mean of prior and OLS variances
        post_var = sigma_ols_sq * gamma_sq / (sigma_ols_sq + gamma_sq + 1e-10)
        beta_logstd_init = np.clip(
            0.5 * np.log(post_var + 1e-10), -5.0, 2.0
        ).astype(np.float32)  # (n_cov, n_genes)

        beta_init = torch.tensor(beta_init_np, dtype=torch.float32)
        beta_logstd_init_t = torch.tensor(beta_logstd_init, dtype=torch.float32)

        # Pre-densify X and state_transitions_t to GPU tensors to eliminate
        # per-batch CPU sparse operations (the dominant training bottleneck).
        # Memory budget: skip if either tensor would exceed 4 GB total.
        x_bytes = m * n * 4
        st_bytes = len(self.unique_edge_indices) * n * 4
        mem_budget = 4 * 1024**3
        # Always keep unique_edge_indices on GPU for fast searchsorted lookups.
        self.unique_edge_indices_t = torch.tensor(
            self.unique_edge_indices, dtype=torch.long
        ).to(self.device)

        if x_bytes + st_bytes <= mem_budget:
            self.x_dense = torch.tensor(self.X.toarray(), dtype=torch.float32).to(
                self.device
            )
            self.st_dense = torch.tensor(
                self.state_transitions_t.toarray(), dtype=torch.float32
            ).to(self.device)
        else:
            self.x_dense = None
            self.st_dense = None

        self.model = SegregVAE(
            n_genes,
            n_covariates,
            hidden_channels=hidden_channels,
            latent_dim=latent_dim,
            kappa=kappa,
            include_diffusion=include_diffusion,
            include_size_factor=include_size_factor,
            log_mean_expr=log_mean_expr,
            beta_init=beta_init,
            beta_logstd_init=beta_logstd_init_t,
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
        kl_annealing: bool = True,
    ):
        if self.include_size_factor:
            self.log_sf_embed = self.log_sf_embed.to(self.device)
            params = list(self.model.parameters()) + list(
                self.log_sf_embed.parameters()
            )
        else:
            params = list(self.model.parameters())
        optimizer = torch.optim.Adam(params, lr=lr)

        loader = NeighborLoader(
            self.data,
            num_neighbors=[nneighbors, nneighbors],
            batch_size=batch_size,
            input_nodes=None,
            shuffle=True,
        )

        self.model.train()

        use_amp = self.device.type == "cuda"
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        scaler = torch.amp.GradScaler(enabled=use_amp)

        pbar = tqdm(range(nepochs), desc="Training Segreg")
        for epoch in pbar:
            # Linear KL annealing: beta_kl increases linearly from 0 up to 50% of nepochs
            if kl_annealing and nepochs > 1:
                current_beta_kl = min(beta_kl, beta_kl * (epoch + 1) / (nepochs // 2))
            else:
                current_beta_kl = beta_kl

            total_loss = 0.0

            for batch in loader:
                optimizer.zero_grad()
                batch = batch.to(self.device)

                # Fetch dense counts for the sampled nodes.
                node_idx = batch.n_id
                if self.x_dense is not None:
                    x_sub_tensor = self.x_dense[node_idx]
                else:
                    x_sub = self.X[node_idx.cpu().numpy()].toarray().astype(np.float32)
                    x_sub_tensor = torch.from_numpy(x_sub).to(self.device)

                if self.model.include_diffusion:
                    # Fetch edge transition weights.
                    global_src = batch.n_id[batch.edge_index[0, :]]
                    global_dst = batch.n_id[batch.edge_index[1, :]]
                    encoded = global_src + global_dst * self.m  # (E,) on GPU

                    mapped_edge_index = torch.searchsorted(
                        self.unique_edge_indices_t, encoded
                    )
                    n_ue = len(self.unique_edge_indices_t)
                    clamped = mapped_edge_index.clamp(max=n_ue - 1)
                    invalid = (mapped_edge_index >= n_ue) | (
                        self.unique_edge_indices_t[clamped] != encoded
                    )
                    if invalid.any():
                        raise RuntimeError(
                            f"Found {invalid.sum()} edges out of {len(encoded)} that are not in unique_edge_indices!"
                        )

                    if self.st_dense is not None:
                        prior_alpha = self.st_dense[mapped_edge_index]
                    else:
                        mapped_np = mapped_edge_index.cpu().numpy()
                        arr = (
                            self.state_transitions_t[mapped_np, :]
                            .toarray()
                            .astype(np.float32)
                        )
                        prior_alpha = torch.from_numpy(arr).to(self.device)
                else:
                    prior_alpha = None

                # Prepare encoder input and run the model forward pass under AMP.
                if self.model.include_size_factor:
                    log_sf = self.log_sf_embed(batch.n_id).squeeze(-1)
                    size_factor = torch.exp(log_sf).unsqueeze(-1)
                    x_norm = x_sub_tensor / (size_factor + 1e-8) * 1000.0
                    encoder_in = torch.log1p(x_norm)
                else:
                    log_sf = None
                    encoder_in = torch.cat([torch.log1p(x_sub_tensor), batch.x], dim=-1)

                with torch.autocast(
                    device_type=self.device.type, dtype=amp_dtype, enabled=use_amp
                ):
                    x_hat, mu, logstd, a, b, alpha, beta = self.model(
                        encoder_in,
                        batch.x,
                        batch.edge_index,
                        prior_alpha,
                        log_size_factor=log_sf,
                    )

                # ── Loss computation in float32 ──────────────────────────────────────
                target_mask = torch.arange(batch.batch_size, device=self.device)
                x_target = x_sub_tensor[target_mask]
                x_hat_target = x_hat[target_mask].float()

                # 1. Reconstruction Loss
                r = F.softplus(self.model.log_r).clamp(min=1e-3)
                mu_nb = x_hat_target
                eps = 1e-8
                log_r_over_r_plus_mu = torch.log(r / (r + mu_nb + eps))
                log_mu_over_r_plus_mu = torch.log((mu_nb + eps) / (r + mu_nb + eps))
                loss_recon = (
                    -(
                        torch.lgamma(x_target + r)
                        - torch.lgamma(r)
                        - torch.lgamma(x_target + 1)
                        + r * log_r_over_r_plus_mu
                        + x_target * log_mu_over_r_plus_mu
                    )
                    .sum(dim=-1)
                    .mean()
                )

                # 2. Node KL Divergence
                mu_target = mu[target_mask].float()
                logstd_target = logstd[target_mask].float()
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

                # 3. Edge KL Divergence
                if self.model.include_diffusion:
                    dst = batch.edge_index[1]
                    edge_mask = dst < batch.batch_size

                    if edge_mask.sum() > 0:
                        a_target = a[edge_mask].float()
                        b_target = b[edge_mask].float()
                        prior_alpha_target = prior_alpha[edge_mask]
                        prior_alpha_target = torch.clamp(
                            prior_alpha_target, 1e-4, 1.0 - 1e-4
                        )
                        kappa = self.model.kappa
                        prior_a = 1.0 + kappa * prior_alpha_target
                        prior_b = 1.0 + kappa * (1.0 - prior_alpha_target)

                        q_alpha = Beta(a_target, b_target)
                        p_alpha = Beta(prior_a, prior_b)
                        kl_alpha = (
                            kl_divergence(q_alpha, p_alpha).sum() / batch.batch_size
                        )
                    else:
                        kl_alpha = torch.tensor(0.0, device=self.device)
                else:
                    kl_alpha = torch.tensor(0.0, device=self.device)

                # 4. Global Regression Parameters KL Divergence — Cauchy prior
                beta_f = beta.float()
                gamma = self.beta_prior_scale_t
                log_q = (
                    -0.5
                    * (
                        (beta_f - self.model.beta_mu)
                        / torch.exp(self.model.beta_logstd)
                    ).pow(2)
                    - self.model.beta_logstd
                    - 0.5 * math.log(2.0 * math.pi)
                )
                log_p = (
                    -math.log(math.pi)
                    - torch.log(gamma)
                    - torch.log1p((beta_f / gamma).pow(2))
                )
                kl_beta = (log_q - log_p).sum() / self.m

                # 5. Size Factor Regularization
                if self.model.include_size_factor:
                    log_sf_target = log_sf[: batch.batch_size]
                    log_sf_prior_target = batch.log_sf_prior[: batch.batch_size]
                    loss_sf = (log_sf_target - log_sf_prior_target).pow(2).mean() / (
                        2 * self.sf_sigma**2
                    )
                else:
                    loss_sf = torch.tensor(0.0, device=self.device)

                loss = loss_recon + current_beta_kl * (kl_z + kl_alpha) + kl_beta + loss_sf

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                loss_val = loss.item()
                if loss_val == loss_val:
                    total_loss += loss_val

            if self.model.include_diffusion:
                kappa_val = self.model.kappa.item()
                kappa_str = f", Kappa: {kappa_val:.2f}"
            else:
                kappa_str = ""

            pbar.set_postfix_str(
                f"Loss: {total_loss / len(loader):.4f} "
                f"(Recon: {loss_recon.item():.4f}, KL_z: {kl_z.item():.4f}, "
                f"KL_b: {kl_beta.item():.4f}{kappa_str})"
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

            # The "minimum credible fold change" is the point in the [Lower, Upper] interval nearest to 0.
            # If the interval contains 0, it is 0.
            # If the interval is entirely positive, it is Lower.
            # If the interval is entirely negative, it is Upper.
            df["MinimumCredible"] = np.where(
                df["Lower"] > 0,
                df["Lower"],
                np.where(df["Upper"] < 0, df["Upper"], 0.0),
            )

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

                if self.model.include_size_factor:
                    log_sf = self.log_sf_embed(batch.n_id).squeeze(-1)
                    size_factor = torch.exp(log_sf).unsqueeze(-1)
                    x_norm = x_sub_tensor / (size_factor + 1e-8) * 1000.0
                    encoder_in = torch.log1p(x_norm)
                else:
                    log_sf = None
                    encoder_in = torch.cat([torch.log1p(x_sub_tensor), batch.x], dim=-1)

                # Forward pass through encoder
                mu, logstd = self.model.encoder(encoder_in, batch.edge_index)

                # PyG places the target nodes first in the batch up to `batch.batch_size`
                target_mask = slice(0, batch.batch_size)
                mu_target = mu[target_mask]
                logstd_target = logstd[target_mask]
                covariates_target = batch.x[target_mask]
                log_sf_target = log_sf[target_mask] if log_sf is not None else None
                beta = self.model.beta_mu

                std_target = torch.exp(logstd_target)

                lam_target_sum = torch.zeros(
                    (batch.batch_size, self.n), device=self.device
                )

                # Monte Carlo sampling to compute expected rates.
                # The size factor is included so the output is on the same scale as raw counts.
                for _ in range(n_samples):
                    z_target = mu_target + torch.randn_like(std_target) * std_target
                    rho_target = self.model.node_decoder(z_target)

                    log_rate = (
                        rho_target + covariates_target @ beta + self.model.gene_bias
                    )
                    if self.model.include_size_factor:
                        log_rate = log_rate + log_sf_target.unsqueeze(-1)
                    lam_target_sum += torch.exp(
                        torch.clamp(log_rate, min=-15.0, max=15.0)
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

    def get_latent_representation(
        self,
        batch_size: int = 4096,
        nneighbors: int = 10,
    ) -> np.ndarray:
        """
        Returns the posterior mean (mu) for each cell in the latent space.
        """
        self.model.eval()

        loader = NeighborLoader(
            self.data,
            num_neighbors=[nneighbors, nneighbors],
            batch_size=batch_size,
            input_nodes=None,
            shuffle=False,
        )

        all_mu = []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)

                # Fetch sparse counts for the sampled nodes and convert to dense tensor
                node_idx = batch.n_id.cpu().numpy()
                x_sub = self.X[node_idx].toarray()
                x_sub_tensor = torch.tensor(
                    x_sub, dtype=torch.float32, device=self.device
                )

                if self.model.include_size_factor:
                    log_sf = self.log_sf_embed(batch.n_id).squeeze(-1)
                    size_factor = torch.exp(log_sf).unsqueeze(-1)
                    x_norm = x_sub_tensor / (size_factor + 1e-8) * 1000.0
                    encoder_in = torch.log1p(x_norm)
                else:
                    log_sf = None
                    encoder_in = torch.cat([torch.log1p(x_sub_tensor), batch.x], dim=-1)

                # Forward pass through encoder
                mu, _ = self.model.encoder(encoder_in, batch.edge_index)

                # PyG places the target nodes first in the batch up to `batch.batch_size`
                target_mask = slice(0, batch.batch_size)
                mu_target = mu[target_mask]

                all_mu.append(mu_target.cpu().numpy())

        return np.concatenate(all_mu, axis=0)
