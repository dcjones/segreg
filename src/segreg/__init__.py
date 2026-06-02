# Radically simplified diffusion model based on cell-purity priors.
import math
import sys
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
from tqdm import tqdm


class Encoder(nn.Module):
    """
    Node-based Encoder with LayerNorm.
    Takes gene counts and design covariates,
    and outputs parameters for the latent normal distribution (mu, logstd).
    """

    def __init__(
        self, in_channels: int, hidden_channels: int = 128, latent_dim: int = 64
    ):
        super().__init__()
        # Initial transformation to latent-like space, using 2 layers for more capacity
        self.lin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
        )

        # Separate heads for mu and logstd
        self.mu_head = nn.Linear(hidden_channels, latent_dim)
        self.logstd_head = nn.Linear(hidden_channels, latent_dim)

    def forward(self, x):
        h = self.lin(x)
        mu = self.mu_head(h)
        logstd = self.logstd_head(h)
        return mu, logstd


class NodeDecoder(nn.Module):
    """
    Decodes the node latent representation back into unconstrained expression rates (rho).
    Uses a simple linear transformation to capture residual biological variation
    as linear gene modules.
    """

    def __init__(self, latent_dim: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.lin = nn.Linear(latent_dim, out_channels, bias=False)

    def forward(self, z):
        return self.lin(z)


class SegregBase(nn.Module):
    """Shared VAE backbone: encoder, decoder, gene_bias, log_r, size factor, inflow scale."""

    def __init__(
        self,
        n_cells: int,
        n_genes: int,
        encoder_in_channels: int,
        hidden_channels: int = 64,
        latent_dim: int = 32,
        rate_offset: float = 1e-2,
        include_diffusion: bool = True,
        include_size_factor: bool = True,
        log_mean_expr: torch.Tensor | None = None,
        log_sf_prior: torch.Tensor | None = None,
    ):
        super().__init__()
        self.include_diffusion = include_diffusion
        self.include_size_factor = include_size_factor
        self.rate_offset = rate_offset

        self.encoder = Encoder(encoder_in_channels, hidden_channels, latent_dim)
        self.node_decoder = NodeDecoder(latent_dim, hidden_channels, n_genes)

        if self.include_size_factor:
            self.log_sf_embed = nn.Embedding(n_cells, 1)
            if log_sf_prior is not None:
                self.log_sf_embed.weight.data = log_sf_prior.clone().unsqueeze(-1)

        if log_mean_expr is not None:
            self.gene_bias = nn.Parameter(log_mean_expr.clone())
        else:
            self.gene_bias = nn.Parameter(torch.zeros(n_genes))

        self.log_r = nn.Parameter(torch.full((n_genes,), 9.3))

        if include_diffusion:
            # Per-gene learnable inflow scale: proseg inflow may systematically underestimate
            # contamination for highly expressed neighboring-cell genes. A learned scale
            # is identifiable because contamination genes are under-fitted (x_hat < x),
            # while genuine DE genes are already well-explained by beta/z and stay near 1.
            self.log_inflow_scale = nn.Parameter(torch.zeros(n_genes))

    def reparameterize(self, mu, logstd):
        if self.training:
            std = torch.exp(logstd.clamp(max=8.0))
            eps = torch.randn_like(std)
            return (mu + eps * std).clamp(-40.0, 40.0)
        return mu.clamp(-40.0, 40.0)

    def prepare_encoder_input(
        self, x_sub_tensor: torch.Tensor, batch_idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Normalize counts and compute size factor. Returns (encoder_in, log_sf)."""
        if self.include_size_factor:
            log_sf = self.log_sf_embed(batch_idx).squeeze(-1)
            x_norm = x_sub_tensor / (torch.exp(log_sf).unsqueeze(-1) + 1e-8) * 1000.0
            encoder_in = torch.log1p(x_norm)
        else:
            log_sf = None
            encoder_in = torch.log1p(x_sub_tensor)
        return encoder_in, log_sf


class SegregVAE(SegregBase):
    def __init__(
        self,
        n_cells: int,
        n_genes: int,
        n_covariates: int,
        hidden_channels: int = 64,
        latent_dim: int = 32,
        kappa: float = 100.0,
        rate_offset: float = 1e-2,
        include_diffusion: bool = True,
        include_size_factor: bool = True,
        log_mean_expr: torch.Tensor | None = None,
        log_sf_prior: torch.Tensor | None = None,
        beta_init: torch.Tensor | None = None,
        beta_logstd_init: torch.Tensor | None = None,
    ):
        encoder_in_channels = n_genes if include_size_factor else n_genes + n_covariates
        super().__init__(
            n_cells=n_cells,
            n_genes=n_genes,
            encoder_in_channels=encoder_in_channels,
            hidden_channels=hidden_channels,
            latent_dim=latent_dim,
            rate_offset=rate_offset,
            include_diffusion=include_diffusion,
            include_size_factor=include_size_factor,
            log_mean_expr=log_mean_expr,
            log_sf_prior=log_sf_prior,
        )

        if beta_init is not None:
            self.beta_mu = nn.Parameter(beta_init.clone())
        else:
            self.beta_mu = nn.Parameter(torch.zeros(n_covariates, n_genes))
        if beta_logstd_init is not None:
            self.beta_logstd = nn.Parameter(beta_logstd_init.clone())
        else:
            self.beta_logstd = nn.Parameter(torch.full((n_covariates, n_genes), -3.0))

    def forward(self, x, covariates, inflow, outflow, log_size_factor=None):
        mu, logstd = self.encoder(x)
        z = self.reparameterize(mu, logstd)
        rho = self.node_decoder(z)

        if self.training:
            beta_std = torch.exp(self.beta_logstd.clamp(max=4.0))
            beta = self.beta_mu + torch.randn_like(beta_std) * beta_std
        else:
            beta = self.beta_mu

        log_rate = rho + covariates @ beta + self.gene_bias
        if self.include_size_factor and log_size_factor is not None:
            log_rate = log_rate + log_size_factor.unsqueeze(-1)
        lam = torch.exp(torch.clamp(log_rate, min=-15.0, max=15.0))

        if self.include_diffusion:
            inflow_scale = torch.exp(self.log_inflow_scale)
            x_hat = lam + inflow_scale * inflow + self.rate_offset
        else:
            x_hat = lam + self.rate_offset

        return x_hat, mu, logstd, beta


class SegregFactorizationVAE(SegregBase):
    def __init__(
        self,
        n_cells: int,
        n_genes: int,
        n_factors: int,
        hidden_channels: int = 64,
        latent_dim: int = 32,
        rate_offset: float = 1e-2,
        include_diffusion: bool = True,
        include_size_factor: bool = True,
        log_mean_expr: torch.Tensor | None = None,
        log_sf_prior: torch.Tensor | None = None,
    ):
        # W is per-cell and looked up by index, not fed to the encoder
        super().__init__(
            n_cells=n_cells,
            n_genes=n_genes,
            encoder_in_channels=n_genes,
            hidden_channels=hidden_channels,
            latent_dim=latent_dim,
            rate_offset=rate_offset,
            include_diffusion=include_diffusion,
            include_size_factor=include_size_factor,
            log_mean_expr=log_mean_expr,
            log_sf_prior=log_sf_prior,
        )
        self.n_factors = n_factors
        self.W_embed = nn.Embedding(n_cells, n_factors)
        nn.init.normal_(self.W_embed.weight, std=0.1)
        self.H = nn.Parameter(torch.randn(n_factors, n_genes) * 0.01)

    def forward(
        self,
        encoder_in: torch.Tensor,
        batch_idx: torch.Tensor,
        inflow: torch.Tensor | None,
        outflow: torch.Tensor | None,
        log_size_factor: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logstd = self.encoder(encoder_in)
        z = self.reparameterize(mu, logstd)
        rho = self.node_decoder(z)

        W = self.W_embed(batch_idx)  # (batch, n_factors)
        log_rate = rho + W @ self.H + self.gene_bias
        if self.include_size_factor and log_size_factor is not None:
            log_rate = log_rate + log_size_factor.unsqueeze(-1)
        lam = torch.exp(torch.clamp(log_rate, min=-15.0, max=15.0))

        if self.include_diffusion:
            inflow_scale = torch.exp(self.log_inflow_scale)
            x_hat = lam + inflow_scale * inflow + self.rate_offset
        else:
            x_hat = lam + self.rate_offset

        return x_hat, mu, logstd, W


def _nb_loss(x_sub_tensor, x_hat, log_r):
    """Negative binomial reconstruction loss, summed over genes, averaged over cells."""
    r = F.softplus(log_r).clamp(min=1e-3)
    mu_nb = x_hat.float()
    eps = 1e-8
    log_r_over_r_plus_mu = torch.log(r / (r + mu_nb + eps))
    log_mu_over_r_plus_mu = torch.log((mu_nb + eps) / (r + mu_nb + eps))
    return (
        -(
            torch.lgamma(x_sub_tensor + r)
            - torch.lgamma(r)
            - torch.lgamma(x_sub_tensor + 1)
            + r * log_r_over_r_plus_mu
            + x_sub_tensor * log_mu_over_r_plus_mu
        )
        .sum(dim=-1)
        .mean()
    )


def _kl_z(mu, logstd):
    """KL divergence from N(mu, exp(logstd)^2) to N(0, 1), averaged over cells."""
    return -0.5 * torch.sum(1 + 2 * logstd - mu.pow(2) - (2 * logstd).exp(), dim=-1).mean()


def _size_factor_loss(log_sf, batch_log_sf_prior, sf_sigma):
    return (log_sf - batch_log_sf_prior).pow(2).mean() / (2 * sf_sigma**2)


def _inflow_scale_loss(log_inflow_scale, inflow_scale_reg):
    return inflow_scale_reg * log_inflow_scale.pow(2).mean()


class SegregTrainingWrapper(nn.Module):
    def __init__(
        self,
        model: SegregVAE,
        beta_prior_scale: torch.Tensor,
        m: int,
        sf_sigma: float,
        inflow_scale_reg: float = 0.1,
    ):
        super().__init__()
        self.model = model
        self.register_buffer("beta_prior_scale", beta_prior_scale)
        self.m = m
        self.sf_sigma = sf_sigma
        self.inflow_scale_reg = inflow_scale_reg

    def forward(
        self,
        batch_idx,
        batch_x,
        batch_log_sf_prior,
        x_sub_tensor,
        inflow_sub,
        outflow_sub,
        current_beta_kl: torch.Tensor,
    ):
        encoder_in, log_sf = self.model.prepare_encoder_input(x_sub_tensor, batch_idx)

        x_hat, mu, logstd, beta = self.model(
            encoder_in,
            batch_x,
            inflow=inflow_sub,
            outflow=outflow_sub,
            log_size_factor=log_sf,
        )

        loss_recon = _nb_loss(x_sub_tensor, x_hat, self.model.log_r)
        kl_z = _kl_z(mu, logstd)

        beta_f = beta.float()
        gamma = self.beta_prior_scale
        log_q = (
            -0.5
            * ((beta_f - self.model.beta_mu) / torch.exp(self.model.beta_logstd)).pow(2)
            - self.model.beta_logstd
            - 0.5 * math.log(2.0 * math.pi)
        )
        log_p = (
            -math.log(math.pi) - torch.log(gamma) - torch.log1p((beta_f / gamma).pow(2))
        )
        kl_beta = (log_q - log_p).sum() / self.m

        if self.model.include_size_factor and log_sf is not None:
            loss_sf = _size_factor_loss(log_sf, batch_log_sf_prior, self.sf_sigma)
        else:
            loss_sf = torch.tensor(0.0, device=x_hat.device)

        if self.model.include_diffusion:
            loss_inflow_scale = _inflow_scale_loss(
                self.model.log_inflow_scale, self.inflow_scale_reg
            )
        else:
            loss_inflow_scale = torch.tensor(0.0, device=x_hat.device)

        loss = loss_recon + (current_beta_kl * kl_z) + kl_beta + loss_sf + loss_inflow_scale
        return loss, loss_recon


class FactorizationTrainingWrapper(nn.Module):
    def __init__(
        self,
        model: SegregFactorizationVAE,
        m: int,
        sf_sigma: float,
        w_reg: float = 1.0,
        h_reg: float = 1.0,
        inflow_scale_reg: float = 0.1,
    ):
        super().__init__()
        self.model = model
        self.m = m
        self.sf_sigma = sf_sigma
        self.w_reg = w_reg
        self.h_reg = h_reg
        self.inflow_scale_reg = inflow_scale_reg

    def forward(
        self,
        batch_idx,
        batch_log_sf_prior,
        x_sub_tensor,
        inflow_sub,
        outflow_sub,
        current_kl_weight: torch.Tensor,
    ):
        encoder_in, log_sf = self.model.prepare_encoder_input(x_sub_tensor, batch_idx)

        x_hat, mu, logstd, W = self.model(
            encoder_in,
            batch_idx,
            inflow=inflow_sub,
            outflow=outflow_sub,
            log_size_factor=log_sf,
        )

        loss_recon = _nb_loss(x_sub_tensor, x_hat, self.model.log_r)
        kl_z = _kl_z(mu, logstd)

        loss_w = self.w_reg * W.pow(2).mean()
        loss_h = self.h_reg * self.model.H.pow(2).mean()

        if self.model.include_size_factor and log_sf is not None:
            loss_sf = _size_factor_loss(log_sf, batch_log_sf_prior, self.sf_sigma)
        else:
            loss_sf = torch.tensor(0.0, device=x_hat.device)

        if self.model.include_diffusion:
            loss_inflow_scale = _inflow_scale_loss(
                self.model.log_inflow_scale, self.inflow_scale_reg
            )
        else:
            loss_inflow_scale = torch.tensor(0.0, device=x_hat.device)

        loss = loss_recon + current_kl_weight * kl_z + loss_w + loss_h + loss_sf + loss_inflow_scale
        return loss, loss_recon


def _ols_init_beta(
    X: csr_matrix,
    inflow: csr_matrix,
    design_np: np.ndarray,
    sf_col: np.ndarray,
    log_mean_expr: np.ndarray,
    covariate_names: list,
    m: int,
    n: int,
    beta_prior_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OLS-based initialization for beta, beta_prior_scale_matrix, and beta_logstd.

    Processes genes in chunks to avoid materializing the full dense m×n matrix.
    Returns (beta_init_np, beta_prior_scale_matrix, beta_logstd_init).
    """
    mean_rate_1d = np.asarray(X.mean(axis=0)).squeeze().astype(np.float32) / float(
        sf_col.mean()
    )
    gene_pseudocount = (0.5 * np.maximum(mean_rate_1d, 2e-8)).reshape(1, -1)

    sf_col_f64 = sf_col.astype(np.float64)
    chunk_size = 1024
    beta_chunks = []
    for g_start in range(0, n, chunk_size):
        g_end = min(g_start + chunk_size, n)
        X_chunk = X[:, g_start:g_end].toarray().astype(np.float64)
        pc = gene_pseudocount[:, g_start:g_end].astype(np.float64)
        y_log = np.log(X_chunk / sf_col_f64 + pc)
        y_resid = y_log - log_mean_expr[g_start:g_end].astype(np.float64)
        beta_chunk, _, _, _ = np.linalg.lstsq(design_np, y_resid, rcond=None)
        beta_chunks.append(beta_chunk.astype(np.float32))
    beta_init_np = np.concatenate(beta_chunks, axis=1)

    scale_per_cov = np.array(
        [10.0 if name == "Intercept" else beta_prior_scale for name in covariate_names],
        dtype=np.float32,
    )
    expressed_rates = mean_rate_1d[mean_rate_1d > 1e-6]
    median_rate = (
        float(np.median(expressed_rates)) if len(expressed_rates) > 0 else 1e-4
    )
    gene_scale = (
        np.sqrt(np.maximum(mean_rate_1d, 1e-8) / median_rate)
        .clip(0.1, 10.0)
        .astype(np.float32)
    )

    # Power-Balanced Prior: scale interaction prior width by 1/sqrt(m) to maintain
    # consistent discipline as statistical power to fit spurious betas grows with m.
    interaction_base_scale = 10.0 / np.sqrt(m)

    # Suspect-Specific Shrinkage: genes where inflow dominates have high contamination
    # potential and should be shrunk more aggressively.
    avg_obs = np.asarray(X.mean(axis=0)).squeeze()
    avg_inflow = np.asarray(inflow.mean(axis=0)).squeeze()
    suspect_score = avg_inflow / (avg_obs + 1e-8)
    suspect_shrinkage = 1.0 / (1.0 + 50.0 * suspect_score)

    beta_prior_scale_matrix = np.zeros((len(covariate_names), n), dtype=np.float32)
    for i, name in enumerate(covariate_names):
        if ":" in name:
            beta_prior_scale_matrix[i, :] = (
                interaction_base_scale * suspect_shrinkage * gene_scale
            )
        else:
            beta_prior_scale_matrix[i, :] = scale_per_cov[i] * gene_scale

    r_init = float(np.log1p(np.exp(9.3)))
    fisher_per_cell_g = r_init * mean_rate_1d / (r_init + mean_rate_1d + 1e-10)
    design_cov_var = np.mean(design_np**2, axis=0).astype(np.float32)
    fisher_info = np.outer(m * design_cov_var, fisher_per_cell_g)
    gamma_sq = beta_prior_scale_matrix**2
    sigma_ols_sq = 1.0 / np.maximum(fisher_info, 1e-6)
    shrink = gamma_sq / (gamma_sq + sigma_ols_sq)
    beta_init_np = beta_init_np * shrink

    for i, name in enumerate(covariate_names):
        if ":" in name:
            beta_init_np[i, :] = 0.0

    post_var = sigma_ols_sq * gamma_sq / (sigma_ols_sq + gamma_sq + 1e-10)
    beta_logstd_init = np.clip(0.5 * np.log(post_var + 1e-10), -5.0, 2.0).astype(
        np.float32
    )

    return beta_init_np, beta_prior_scale_matrix, beta_logstd_init


def _load_proseg_data(
    data: SpatialData | AnnData,
) -> tuple[AnnData, csr_matrix, csr_matrix, csr_matrix, np.ndarray]:
    """Extract expression, inflow, outflow, and size factors from a proseg AnnData/SpatialData."""
    if isinstance(data, AnnData):
        adata = data
    elif isinstance(data, SpatialData):
        adata = data.tables["table"]
    else:
        raise ValueError("data must be an AnnData or SpatialData object")

    if "proseg_run" not in adata.uns:
        raise ValueError("This is not a proseg spatialdata file")

    X = adata.X.tocsr() if not isinstance(adata.X, csr_matrix) else adata.X
    inflow = adata.layers["expected_inflow"].tocsr()
    assert isinstance(inflow, csr_matrix)
    outflow = adata.layers["expected_outflow"].tocsr()
    assert isinstance(outflow, csr_matrix)

    return adata, X, inflow, outflow


class RegressionModel:
    X: csr_matrix
    inflow: csr_matrix | None
    outflow: csr_matrix | None
    data: Data
    design: DesignMatrix
    device: torch.device
    model: SegregVAE
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
        rate_offset: float = 1e-2,
        hidden_channels: int = 128,
        latent_dim: int = 64,
        kappa: float = 1000.0,
        beta_prior_scale: float = 1.0,
        inflow_scale_reg: float = 1.0,
    ):
        adata, self.X, self.inflow, self.outflow = _load_proseg_data(data)

        self.m, self.n = adata.shape
        self.var_names = adata.var_names
        self.obs_names = adata.obs_names

        design_df = dmatrix(formula, adata.obs, return_type="dataframe")
        self.design = cast(DesignMatrix, design_df)

        if design_df.shape[0] < self.m:
            design_df = design_df.reindex(adata.obs.index, fill_value=0.0)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if include_size_factor and "volume" in adata.obs.columns:
            cell_size = np.asarray(adata.obs["volume"]).squeeze().astype(np.float64)
        else:
            cell_size = np.asarray(self.X.sum(axis=1)).squeeze().astype(np.float64)
        log_size_factors = np.log(cell_size + 1e-8).astype(np.float32)

        self.data = Data(
            n_id=torch.arange(self.m),
            x=torch.tensor(np.asarray(design_df), dtype=torch.float32),
            log_sf_prior=torch.tensor(log_size_factors, dtype=torch.float32),
        )

        self.sf_sigma = sf_sigma
        self.inflow_scale_reg = inflow_scale_reg
        self.include_size_factor = include_size_factor

        sf_col = np.exp(log_size_factors).reshape(-1, 1)
        mean_expr_raw = np.asarray(self.X.mean(axis=0)).squeeze().astype(np.float32)
        if include_size_factor:
            log_mean_expr = torch.tensor(
                np.log(mean_expr_raw / float(sf_col.mean()) + 1e-8), dtype=torch.float32
            )
        else:
            log_mean_expr = torch.tensor(
                np.log(mean_expr_raw + 1e-4), dtype=torch.float32
            )

        n_covariates = self.design.shape[1]
        covariate_names = list(self.design.design_info.column_names)
        design_np = np.asarray(design_df, dtype=np.float64)

        beta_init_np, beta_prior_scale_matrix, beta_logstd_init = _ols_init_beta(
            self.X,
            self.inflow,
            design_np,
            sf_col,
            log_mean_expr.numpy(),
            covariate_names,
            self.m,
            self.n,
            beta_prior_scale,
        )

        self.beta_prior_scale_t = torch.tensor(
            beta_prior_scale_matrix, dtype=torch.float32
        ).to(self.device)

        self.model = SegregVAE(
            self.m,
            self.n,
            n_covariates,
            hidden_channels=hidden_channels,
            latent_dim=latent_dim,
            kappa=kappa,
            rate_offset=rate_offset,
            include_diffusion=include_diffusion,
            include_size_factor=include_size_factor,
            log_mean_expr=log_mean_expr,
            log_sf_prior=torch.tensor(log_size_factors, dtype=torch.float32),
            beta_init=torch.tensor(beta_init_np),
            beta_logstd_init=torch.tensor(beta_logstd_init),
        )

    def fit(
        self,
        nepochs: int = 200,
        batch_size: int = 1024,
        lr: float = 1e-3,
        beta_kl: float = 1.0,
        alpha_kl: float = 1.0,
        kl_annealing: bool = True,
        seed: int | None = 42,
        compile: bool = False,
        quiet: bool = False,
    ):
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)

        from torch.utils.data import DataLoader, TensorDataset

        dataset = TensorDataset(
            torch.arange(self.m), self.data.x, self.data.log_sf_prior
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.to(self.device)
        self.model.train()

        training_wrapper = SegregTrainingWrapper(
            self.model,
            self.beta_prior_scale_t,
            self.m,
            self.sf_sigma,
            inflow_scale_reg=self.inflow_scale_reg,
        )

        if compile:
            # torch.compile provides ~20% speedup on CUDA.
            # Requires CUDA toolkit (ptxas) to be in PATH and TRITON_PTXAS_PATH set if not standard.
            print("Compiling model...")
            training_wrapper = torch.compile(training_wrapper, mode="reduce-overhead")

        use_amp = self.device.type == "cuda"
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        scaler = torch.amp.GradScaler(enabled=use_amp)

        pbar = tqdm(range(nepochs), desc="Training Segreg", disable=quiet)
        for epoch in pbar:
            if kl_annealing and nepochs > 1:
                progress = min(1.0, (epoch + 1) / (nepochs // 2))
                current_beta_kl = beta_kl * progress
            else:
                current_beta_kl = beta_kl

            current_beta_kl_t = torch.tensor(current_beta_kl, device=self.device)

            total_loss = 0.0
            for batch_idx, batch_x, batch_log_sf_prior in loader:
                optimizer.zero_grad()
                batch_idx = batch_idx.to(self.device)
                batch_x = batch_x.to(self.device)
                batch_log_sf_prior = batch_log_sf_prior.to(self.device)

                x_sub = self.X[batch_idx.cpu().numpy()].toarray().astype(np.float32)
                x_sub_tensor = torch.from_numpy(x_sub).to(self.device)

                if self.model.include_diffusion:
                    inflow_sub = (
                        self.inflow[batch_idx.cpu().numpy()]
                        .toarray()
                        .astype(np.float32)
                    )
                    outflow_sub = (
                        self.outflow[batch_idx.cpu().numpy()]
                        .toarray()
                        .astype(np.float32)
                    )

                    inflow_sub_tensor = torch.from_numpy(inflow_sub).to(self.device)
                    outflow_sub_tensor = torch.from_numpy(outflow_sub).to(self.device)
                else:
                    inflow_sub_tensor = None
                    outflow_sub_tensor = None

                with torch.autocast(
                    device_type=self.device.type, dtype=amp_dtype, enabled=use_amp
                ):
                    loss, loss_recon = training_wrapper(
                        batch_idx,
                        batch_x,
                        batch_log_sf_prior,
                        x_sub_tensor,
                        inflow_sub_tensor,
                        outflow_sub_tensor,
                        current_beta_kl_t,
                    )

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item()

            pbar.set_description(
                f"Loss: {total_loss / len(loader):.4f} (Recon: {loss_recon:.2f})"
            )

        self.model.eval()

    def get_regression_coefficients(
        self, credible_interval: float | None = None
    ) -> pd.DataFrame:
        beta_mu = self.model.beta_mu.detach().cpu().numpy()
        covariate_names = self.design.design_info.column_names
        df = (
            pd.DataFrame(beta_mu, index=covariate_names, columns=self.var_names)
            .melt(ignore_index=False, var_name="Gene", value_name="Mean")
            .reset_index(names="Covariate")
        )

        # TODO: I'd like to also (optionally) have something like the posterior
        # probability of up- and downregulation.

        if credible_interval is not None:
            beta_std = torch.exp(self.model.beta_logstd).detach().cpu().numpy()
            z = stats.norm.ppf(1.0 - (1.0 - credible_interval) / 2.0)
            df["Lower"] = (beta_mu - z * beta_std).flatten(order="F")
            df["Upper"] = (beta_mu + z * beta_std).flatten(order="F")
            df["MinimumCredible"] = np.where(
                df["Lower"] > 0,
                df["Lower"],
                np.where(df["Upper"] < 0, df["Upper"], 0.0),
            )
        return df

    def get_corrected_expression(
        self, threshold: float = 1e-4, batch_size: int = 4096, n_samples: int = 10
    ) -> csr_matrix:
        self.model.eval()
        from torch.utils.data import DataLoader, TensorDataset

        loader = DataLoader(
            TensorDataset(torch.arange(self.m), self.data.x),
            batch_size=batch_size,
            shuffle=False,
        )
        rows, cols, data = [], [], []
        current_row = 0
        with torch.no_grad():
            for batch_idx, batch_x in loader:
                batch_idx, batch_x = batch_idx.to(self.device), batch_x.to(self.device)
                x_sub_tensor = torch.tensor(
                    self.X[batch_idx.cpu().numpy()].toarray(),
                    dtype=torch.float32,
                    device=self.device,
                )
                encoder_in, log_sf = self.model.prepare_encoder_input(x_sub_tensor, batch_idx)
                mu, logstd = self.model.encoder(encoder_in)
                std = torch.exp(logstd)
                lam_sum = torch.zeros((batch_idx.size(0), self.n), device=self.device)
                for _ in range(n_samples):
                    z = mu + torch.randn_like(std) * std
                    rho = self.model.node_decoder(z)
                    log_rate = rho + batch_x @ self.model.beta_mu + self.model.gene_bias
                    if self.model.include_size_factor and log_sf is not None:
                        log_rate = log_rate + log_sf.unsqueeze(-1)
                    lam = torch.exp(torch.clamp(log_rate, min=-15.0, max=15.0))
                    lam_sum += lam + self.model.rate_offset
                lam_np = (lam_sum / n_samples).cpu().numpy()
                lam_np[lam_np < threshold] = 0.0
                r, c = np.nonzero(lam_np)
                rows.append(r + current_row)
                cols.append(c)
                data.append(lam_np[r, c])
                current_row += batch_idx.size(0)
        return csr_matrix(
            (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
            shape=(self.m, self.n),
        )

    def get_latent_representation(self, batch_size: int = 4096) -> np.ndarray:
        self.model.eval()
        from torch.utils.data import DataLoader, TensorDataset

        loader = DataLoader(
            TensorDataset(torch.arange(self.m), self.data.x),
            batch_size=batch_size,
            shuffle=False,
        )
        all_mu = []
        with torch.no_grad():
            for batch_idx, batch_x in loader:
                batch_idx, batch_x = batch_idx.to(self.device), batch_x.to(self.device)
                x_sub_tensor = torch.tensor(
                    self.X[batch_idx.cpu().numpy()].toarray(),
                    dtype=torch.float32,
                    device=self.device,
                )
                encoder_in, _ = self.model.prepare_encoder_input(x_sub_tensor, batch_idx)
                mu, _ = self.model.encoder(encoder_in)
                all_mu.append(mu.cpu().numpy())
        return np.concatenate(all_mu, axis=0)


class FactorizationModel:
    X: csr_matrix
    inflow: csr_matrix | None
    outflow: csr_matrix | None
    device: torch.device
    model: SegregFactorizationVAE
    m: int
    n: int

    def __init__(
        self,
        data: SpatialData | AnnData,
        n_factors: int,
        batch_size: int | None = 4096,
        include_diffusion: bool = True,
        include_size_factor: bool = True,
        sf_sigma: float = 0.5,
        rate_offset: float = 1e-2,
        hidden_channels: int = 128,
        latent_dim: int = 64,
        w_reg: float = 1.0,
        h_reg: float = 1.0,
        inflow_scale_reg: float = 1.0,
    ):
        adata, self.X, self.inflow, self.outflow = _load_proseg_data(data)

        self.m, self.n = adata.shape
        self.var_names = adata.var_names
        self.obs_names = adata.obs_names

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if include_size_factor and "volume" in adata.obs.columns:
            cell_size = np.asarray(adata.obs["volume"]).squeeze().astype(np.float64)
        else:
            cell_size = np.asarray(self.X.sum(axis=1)).squeeze().astype(np.float64)
        log_size_factors = np.log(cell_size + 1e-8).astype(np.float32)

        self.log_sf_prior_t = torch.tensor(log_size_factors, dtype=torch.float32)

        self.sf_sigma = sf_sigma
        self.w_reg = w_reg
        self.h_reg = h_reg
        self.inflow_scale_reg = inflow_scale_reg

        sf_col = np.exp(log_size_factors).reshape(-1, 1)
        mean_expr_raw = np.asarray(self.X.mean(axis=0)).squeeze().astype(np.float32)
        if include_size_factor:
            log_mean_expr = torch.tensor(
                np.log(mean_expr_raw / float(sf_col.mean()) + 1e-8), dtype=torch.float32
            )
        else:
            log_mean_expr = torch.tensor(
                np.log(mean_expr_raw + 1e-4), dtype=torch.float32
            )

        self.model = SegregFactorizationVAE(
            self.m,
            self.n,
            n_factors,
            hidden_channels=hidden_channels,
            latent_dim=latent_dim,
            rate_offset=rate_offset,
            include_diffusion=include_diffusion,
            include_size_factor=include_size_factor,
            log_mean_expr=log_mean_expr,
            log_sf_prior=torch.tensor(log_size_factors, dtype=torch.float32),
        )

    def fit(
        self,
        nepochs: int = 200,
        batch_size: int = 1024,
        lr: float = 1e-3,
        beta_kl: float = 1.0,
        kl_annealing: bool = True,
        seed: int | None = 42,
        compile: bool = False,
        quiet: bool = False,
    ):
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)

        from torch.utils.data import DataLoader, TensorDataset

        dataset = TensorDataset(torch.arange(self.m), self.log_sf_prior_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.to(self.device)
        self.model.train()

        training_wrapper = FactorizationTrainingWrapper(
            self.model,
            self.m,
            self.sf_sigma,
            w_reg=self.w_reg,
            h_reg=self.h_reg,
            inflow_scale_reg=self.inflow_scale_reg,
        )

        if compile:
            print("Compiling model...")
            training_wrapper = torch.compile(training_wrapper, mode="reduce-overhead")

        use_amp = self.device.type == "cuda"
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        scaler = torch.amp.GradScaler(enabled=use_amp)

        pbar = tqdm(range(nepochs), desc="Training FactorizationModel", disable=quiet)
        for epoch in pbar:
            if kl_annealing and nepochs > 1:
                progress = min(1.0, (epoch + 1) / (nepochs // 2))
                current_kl = beta_kl * progress
            else:
                current_kl = beta_kl

            current_kl_t = torch.tensor(current_kl, device=self.device)

            total_loss = 0.0
            for batch_idx, batch_log_sf_prior in loader:
                optimizer.zero_grad()
                batch_idx = batch_idx.to(self.device)
                batch_log_sf_prior = batch_log_sf_prior.to(self.device)

                x_sub = self.X[batch_idx.cpu().numpy()].toarray().astype(np.float32)
                x_sub_tensor = torch.from_numpy(x_sub).to(self.device)

                if self.model.include_diffusion:
                    inflow_sub = (
                        self.inflow[batch_idx.cpu().numpy()]
                        .toarray()
                        .astype(np.float32)
                    )
                    outflow_sub = (
                        self.outflow[batch_idx.cpu().numpy()]
                        .toarray()
                        .astype(np.float32)
                    )
                    inflow_sub_tensor = torch.from_numpy(inflow_sub).to(self.device)
                    outflow_sub_tensor = torch.from_numpy(outflow_sub).to(self.device)
                else:
                    inflow_sub_tensor = None
                    outflow_sub_tensor = None

                with torch.autocast(
                    device_type=self.device.type, dtype=amp_dtype, enabled=use_amp
                ):
                    loss, loss_recon = training_wrapper(
                        batch_idx,
                        batch_log_sf_prior,
                        x_sub_tensor,
                        inflow_sub_tensor,
                        outflow_sub_tensor,
                        current_kl_t,
                    )

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item()

            pbar.set_description(
                f"Loss: {total_loss / len(loader):.4f} (Recon: {loss_recon:.2f})"
            )

        self.model.eval()

    def get_factor_loadings(self) -> np.ndarray:
        """Returns W matrix, shape (n_cells, n_factors)."""
        return self.model.W_embed.weight.detach().cpu().numpy()

    def get_factor_programs(self) -> np.ndarray:
        """Returns H matrix, shape (n_factors, n_genes)."""
        return self.model.H.detach().cpu().numpy()

    def get_corrected_expression(
        self, threshold: float = 1e-4, batch_size: int = 4096, n_samples: int = 10
    ) -> csr_matrix:
        self.model.eval()
        from torch.utils.data import DataLoader, TensorDataset

        loader = DataLoader(
            TensorDataset(torch.arange(self.m), self.log_sf_prior_t),
            batch_size=batch_size,
            shuffle=False,
        )
        rows, cols, data = [], [], []
        current_row = 0
        with torch.no_grad():
            for batch_idx, _ in loader:
                batch_idx = batch_idx.to(self.device)
                x_sub_tensor = torch.tensor(
                    self.X[batch_idx.cpu().numpy()].toarray(),
                    dtype=torch.float32,
                    device=self.device,
                )
                encoder_in, log_sf = self.model.prepare_encoder_input(x_sub_tensor, batch_idx)
                mu, logstd = self.model.encoder(encoder_in)
                std = torch.exp(logstd)
                lam_sum = torch.zeros((batch_idx.size(0), self.n), device=self.device)
                for _ in range(n_samples):
                    z = mu + torch.randn_like(std) * std
                    rho = self.model.node_decoder(z)
                    W = self.model.W_embed(batch_idx)
                    log_rate = rho + W @ self.model.H + self.model.gene_bias
                    if self.model.include_size_factor and log_sf is not None:
                        log_rate = log_rate + log_sf.unsqueeze(-1)
                    lam = torch.exp(torch.clamp(log_rate, min=-15.0, max=15.0))
                    lam_sum += lam + self.model.rate_offset
                lam_np = (lam_sum / n_samples).cpu().numpy()
                lam_np[lam_np < threshold] = 0.0
                r, c = np.nonzero(lam_np)
                rows.append(r + current_row)
                cols.append(c)
                data.append(lam_np[r, c])
                current_row += batch_idx.size(0)
        return csr_matrix(
            (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
            shape=(self.m, self.n),
        )

    def get_latent_representation(self, batch_size: int = 4096) -> np.ndarray:
        self.model.eval()
        from torch.utils.data import DataLoader, TensorDataset

        loader = DataLoader(
            TensorDataset(torch.arange(self.m), self.log_sf_prior_t),
            batch_size=batch_size,
            shuffle=False,
        )
        all_mu = []
        with torch.no_grad():
            for batch_idx, _ in loader:
                batch_idx = batch_idx.to(self.device)
                x_sub_tensor = torch.tensor(
                    self.X[batch_idx.cpu().numpy()].toarray(),
                    dtype=torch.float32,
                    device=self.device,
                )
                encoder_in, _ = self.model.prepare_encoder_input(x_sub_tensor, batch_idx)
                mu, _ = self.model.encoder(encoder_in)
                all_mu.append(mu.cpu().numpy())
        return np.concatenate(all_mu, axis=0)
