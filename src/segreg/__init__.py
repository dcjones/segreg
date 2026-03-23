# Radically simplified diffusion model based on cell-purity priors.
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


class SegregVAE(nn.Module):
    def __init__(
        self,
        n_cells: int,
        n_genes: int,
        n_covariates: int,
        hidden_channels: int = 64,
        latent_dim: int = 32,
        kappa: float = 100.0,
        include_diffusion: bool = True,
        include_size_factor: bool = True,
        log_mean_expr: torch.Tensor | None = None,
        log_sf_prior: torch.Tensor | None = None,
        beta_init: torch.Tensor | None = None,
        beta_logstd_init: torch.Tensor | None = None,
    ):
        super().__init__()
        self.include_diffusion = include_diffusion
        self.include_size_factor = include_size_factor
        in_channels = n_genes if include_size_factor else n_genes + n_covariates
        self.encoder = Encoder(in_channels, hidden_channels, latent_dim)
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
            # log_alpha initialized so that alpha = 1.0
            # alpha = 0.1 + softplus(log_alpha)
            # 1.0 = 0.1 + softplus(log_alpha) => 0.9 = softplus(log_alpha)
            # log_alpha = log(exp(0.9) - 1) approx 0.37
            self.log_alpha = nn.Parameter(torch.full((n_genes,), 0.3747))

        if beta_init is not None:
            self.beta_mu = nn.Parameter(beta_init.clone())
        else:
            self.beta_mu = nn.Parameter(torch.zeros(n_covariates, n_genes))
        if beta_logstd_init is not None:
            self.beta_logstd = nn.Parameter(beta_logstd_init.clone())
        else:
            self.beta_logstd = nn.Parameter(torch.full((n_covariates, n_genes), -3.0))

    def reparameterize(self, mu, logstd):
        if self.training:
            std = torch.exp(logstd.clamp(max=8.0))
            eps = torch.randn_like(std)
            return (mu + eps * std).clamp(-40.0, 40.0)
        return mu.clamp(-40.0, 40.0)

    def forward(
        self, x, covariates, pi_prior=None, inflow=None, log_size_factor=None
    ):
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

        if self.include_diffusion and pi_prior is not None:
            # Deterministic Simplified Diffusion: fix pi to the prior
            # alpha scales the pre-calculated inflow.
            alpha = 0.1 + F.softplus(self.log_alpha)
            x_hat = pi_prior * lam + alpha * inflow
        else:
            x_hat = lam

        return x_hat, mu, logstd, beta


class SegregTrainingWrapper(nn.Module):
    def __init__(self, model: SegregVAE, beta_prior_scale: torch.Tensor, m: int, sf_sigma: float):
        super().__init__()
        self.model = model
        self.register_buffer("beta_prior_scale", beta_prior_scale)
        self.m = m
        self.sf_sigma = sf_sigma

    def forward(
        self,
        batch_idx,
        batch_x,
        batch_log_sf_prior,
        x_sub_tensor,
        pi_prior_sub,
        inflow_sub,
        current_beta_kl: torch.Tensor,
        current_alpha_kl: torch.Tensor,
    ):
        if self.model.include_size_factor:
            log_sf = self.model.log_sf_embed(batch_idx).squeeze(-1)
            size_factor = torch.exp(log_sf).unsqueeze(-1)
            x_norm = x_sub_tensor / (size_factor + 1e-8) * 1000.0
            encoder_in = torch.log1p(x_norm)
        else:
            log_sf = None
            encoder_in = torch.log1p(x_sub_tensor)

        x_hat, mu, logstd, beta = self.model(
            encoder_in,
            batch_x,
            pi_prior=pi_prior_sub,
            inflow=inflow_sub,
            log_size_factor=log_sf,
        )

        x_hat_target = x_hat.float()
        r = F.softplus(self.model.log_r).clamp(min=1e-3)
        mu_nb = x_hat_target
        eps = 1e-8
        log_r_over_r_plus_mu = torch.log(r / (r + mu_nb + eps))
        log_mu_over_r_plus_mu = torch.log((mu_nb + eps) / (r + mu_nb + eps))
        loss_recon = -(
            torch.lgamma(x_sub_tensor + r)
            - torch.lgamma(r)
            - torch.lgamma(x_sub_tensor + 1)
            + r * log_r_over_r_plus_mu
            + x_sub_tensor * log_mu_over_r_plus_mu
        ).sum(dim=-1).mean()

        kl_z = -0.5 * torch.sum(
            1 + 2 * logstd - mu.pow(2) - (2 * logstd).exp(), dim=-1
        ).mean()

        beta_f = beta.float()
        gamma = self.beta_prior_scale
        log_q = (
            -0.5 * ((beta_f - self.model.beta_mu) / torch.exp(self.model.beta_logstd)).pow(2)
            - self.model.beta_logstd
            - 0.5 * math.log(2.0 * math.pi)
        )
        log_p = (
            -math.log(math.pi)
            - torch.log(gamma)
            - torch.log1p((beta_f / gamma).pow(2))
        )
        kl_beta = (log_q - log_p).sum() / self.m

        if self.model.include_size_factor:
            loss_sf = (
                (log_sf - batch_log_sf_prior).pow(2).mean() / (2 * self.sf_sigma**2)
            )
        else:
            loss_sf = torch.tensor(0.0, device=x_hat.device)

        if self.model.include_diffusion:
            # log_alpha prior: N(0.3747, 0.5) to keep alpha near 1.0
            kl_alpha = 0.5 * ((self.model.log_alpha - 0.3747)**2 / (0.5**2)).sum() / self.m
        else:
            kl_alpha = torch.tensor(0.0, device=x_hat.device)

        loss = loss_recon + (current_beta_kl * kl_z) + kl_beta + loss_sf + (current_alpha_kl * kl_alpha)
        return loss, loss_recon


class RegressionModel:
    X: csr_matrix
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
        hidden_channels: int = 128,
        latent_dim: int = 64,
        kappa: float = 1000.0,
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

        self.m, self.n = adata.shape
        self.var_names = adata.var_names
        self.obs_names = adata.obs_names

        design_df = dmatrix(formula, adata.obs, return_type="dataframe")
        self.design = cast(DesignMatrix, design_df)

        if design_df.shape[0] < self.m:
            design_df = design_df.reindex(adata.obs.index, fill_value=0.0)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.X = adata.X.tocsr() if not isinstance(adata.X, csr_matrix) else adata.X
        X_dense = self.X.toarray().astype(np.float32)

        # This is a flattened 3d array giving per-gene state transition probabilities
        state_transitions = adata.varm["state_transitions"]

        # SciPy/AnnData zarr loaders sometimes use int32 for indices, which overflows if m*m > 2.14 billion.
        # CSR row indices must be monotonically increasing, so we can detect and fix negative wraps.
        if state_transitions.indices.dtype == np.int32 and self.m * self.m > 2147483647:
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
                shape=(self.n, self.m * self.m),
            )

        # Simplified Diffusion Pre-calculations:
        print("Pre-calculating simplified diffusion terms...")
        # We use a Direct Difference model:
        # ExpectedObserved = lambda + Contamination
        # where Contamination = max(0, Observed - ExpectedTrue)
        # This forces lambda to fit the 'cleaned' counts (ExpectedTrue).
        
        inflow = np.zeros((self.m, self.n), dtype=np.float32)
        
        for g in range(self.n):
            # Row g of state_transitions gives ST_{ij}^g for all i,j
            # ST is (n_genes, m*m), index = src_true + dst_obs * m
            # We construct a sparse (m, m) matrix S_g where S_g[dst_obs, src_true] = P(true | obs)
            row = state_transitions[g, :]
            S_g = csr_matrix(
                (row.data, (row.indices // self.m, row.indices % self.m)), 
                shape=(self.m, self.m)
            )
            
            counts_g = X_dense[:, g]
            # Purity-based inflow: remove all expected contamination.
            # This is more robust than net inflow (C-T) because it doesn't
            # allow local production to 'cancel out' incoming contamination.
            p_self = S_g.diagonal()
            inflow[:, g] = (1.0 - p_self) * counts_g
            
        self.inflow_t = torch.tensor(inflow, dtype=torch.float32).to(self.device)
        # pi is fixed to 1.0 in this model
        self.pi_prior_t = torch.ones((self.m, self.n), dtype=torch.float32).to(self.device)
        print("Done.")

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
        self.include_size_factor = include_size_factor

        mean_expr = torch.tensor(X_dense.mean(axis=0), dtype=torch.float32)
        if include_size_factor:
            mean_sf = float(cell_size.mean())
            log_mean_expr = torch.log(mean_expr / mean_sf + 1e-8)
        else:
            log_mean_expr = torch.log(mean_expr + 1e-4)

        n_covariates = self.design.shape[1]
        print("Computing OLS initialization for beta...")
        sf_col = np.exp(log_size_factors).reshape(-1, 1)
        mean_rate = np.asarray(X_dense.mean(axis=0)).squeeze().astype(np.float32).reshape(1, -1) / float(sf_col.mean())
        gene_pseudocount = 0.5 * np.maximum(mean_rate, 2e-8)
        y_log = np.log(X_dense / sf_col + gene_pseudocount)
        y_resid = y_log - log_mean_expr.numpy()
        design_np = np.asarray(design_df, dtype=np.float64)
        beta_init_np, _, _, _ = np.linalg.lstsq(design_np, y_resid.astype(np.float64), rcond=None)
        beta_init_np = beta_init_np.astype(np.float32)
        print("Done.")

        covariate_names = list(self.design.design_info.column_names)
        scale_per_cov = np.array([10.0 if name == "Intercept" else beta_prior_scale for name in covariate_names], dtype=np.float32)
        mean_rate_1d = mean_rate.squeeze()
        expressed_rates = mean_rate_1d[mean_rate_1d > 1e-6]
        median_rate = float(np.median(expressed_rates)) if len(expressed_rates) > 0 else 1e-4
        gene_scale = np.sqrt(np.maximum(mean_rate_1d, 1e-8) / median_rate).clip(0.1, 10.0).astype(np.float32)
        
        # Power-Balanced Prior: Interaction terms represent local deviations.
        # As m increases, statistical power to fit spurious non-zero betas grows.
        # We scale the interaction prior width by 1/sqrt(m) to maintain consistent discipline.
        interaction_base_scale = 20.0 / np.sqrt(self.m)
        
        # Suspect-Specific Shrinkage: calculate contamination potential for each gene.
        # genes where P(self|obs) is low have high contamination potential.
        avg_obs = X_dense.mean(axis=0)
        avg_exp_true = avg_obs - inflow.mean(axis=0)
        # Suspect score: ratio of contamination to total signal
        suspect_score = (avg_obs - avg_exp_true) / (avg_obs + 1e-8)
        # f(s) = 1 / (1 + 50 * s)
        suspect_shrinkage = 1.0 / (1.0 + 50.0 * suspect_score)

        beta_prior_scale_matrix = np.zeros((len(covariate_names), self.n), dtype=np.float32)
        for i, name in enumerate(covariate_names):
            if ":" in name:
                # Interaction terms get power-balanced scale and suspect shrinkage
                beta_prior_scale_matrix[i, :] = interaction_base_scale * suspect_shrinkage
            else:
                beta_prior_scale_matrix[i, :] = scale_per_cov[i] * gene_scale
        
        self.beta_prior_scale_t = torch.tensor(beta_prior_scale_matrix, dtype=torch.float32).to(self.device)

        r_init = float(np.log1p(np.exp(9.3)))
        fisher_per_cell_g = r_init * mean_rate_1d / (r_init + mean_rate_1d + 1e-10)
        design_cov_var = np.mean(design_np**2, axis=0).astype(np.float32)
        fisher_info = np.outer(self.m * design_cov_var, fisher_per_cell_g)
        gamma_sq = beta_prior_scale_matrix**2
        sigma_ols_sq = 1.0 / np.maximum(fisher_info, 1e-6)
        shrink = gamma_sq / (gamma_sq + sigma_ols_sq)
        beta_init_np = beta_init_np * shrink

        for i, name in enumerate(covariate_names):
            if ":" in name:
                beta_init_np[i, :] = 0.0

        post_var = sigma_ols_sq * gamma_sq / (sigma_ols_sq + gamma_sq + 1e-10)
        beta_logstd_init = np.clip(0.5 * np.log(post_var + 1e-10), -5.0, 2.0).astype(np.float32)

        self.model = SegregVAE(
            self.m,
            self.n,
            n_covariates,
            hidden_channels=hidden_channels,
            latent_dim=latent_dim,
            kappa=kappa,
            include_diffusion=include_diffusion,
            include_size_factor=include_size_factor,
            log_mean_expr=log_mean_expr,
            log_sf_prior=torch.tensor(log_size_factors, dtype=torch.float32),
            beta_init=torch.tensor(beta_init_np),
            beta_logstd_init=torch.tensor(beta_logstd_init),
        )

        x_bytes = self.m * self.n * 4
        mem_budget = 4 * 1024**3
        if x_bytes <= mem_budget:
            self.x_dense = torch.tensor(X_dense, dtype=torch.float32).to(self.device)
        else:
            self.x_dense = None

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
    ):
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)

        from torch.utils.data import DataLoader, TensorDataset
        dataset = TensorDataset(
            torch.arange(self.m),
            self.data.x,
            self.data.log_sf_prior
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.to(self.device)
        self.model.train()

        training_wrapper = SegregTrainingWrapper(
            self.model, self.beta_prior_scale_t, self.m, self.sf_sigma
        )

        if compile:
            # torch.compile provides ~20% speedup on CUDA.
            # Requires CUDA toolkit (ptxas) to be in PATH and TRITON_PTXAS_PATH set if not standard.
            print("Compiling model...")
            training_wrapper = torch.compile(training_wrapper, mode="reduce-overhead")

        use_amp = self.device.type == "cuda"
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        scaler = torch.amp.GradScaler(enabled=use_amp)

        pbar = tqdm(range(nepochs), desc="Training Segreg")
        for epoch in pbar:
            if kl_annealing and nepochs > 1:
                progress = min(1.0, (epoch + 1) / (nepochs // 2))
                current_beta_kl = beta_kl * progress
                current_alpha_kl = alpha_kl * progress
            else:
                current_beta_kl = beta_kl
                current_alpha_kl = alpha_kl

            current_beta_kl_t = torch.tensor(current_beta_kl, device=self.device)
            current_alpha_kl_t = torch.tensor(current_alpha_kl, device=self.device)

            total_loss = 0.0
            for batch_idx, batch_x, batch_log_sf_prior in loader:
                optimizer.zero_grad()
                batch_idx = batch_idx.to(self.device)
                batch_x = batch_x.to(self.device)
                batch_log_sf_prior = batch_log_sf_prior.to(self.device)

                if self.x_dense is not None:
                    x_sub_tensor = self.x_dense[batch_idx]
                else:
                    x_sub = self.X[batch_idx.cpu().numpy()].toarray().astype(np.float32)
                    x_sub_tensor = torch.from_numpy(x_sub).to(self.device)

                if self.model.include_diffusion:
                    pi_prior_sub = self.pi_prior_t[batch_idx]
                    inflow_sub = self.inflow_t[batch_idx]
                else:
                    pi_prior_sub = inflow_sub = None

                with torch.autocast(
                    device_type=self.device.type, dtype=amp_dtype, enabled=use_amp
                ):
                    loss, loss_recon = training_wrapper(
                        batch_idx,
                        batch_x,
                        batch_log_sf_prior,
                        x_sub_tensor,
                        pi_prior_sub,
                        inflow_sub,
                        current_beta_kl_t,
                        current_alpha_kl_t,
                    )

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item()

            pbar.set_description(
                f"Loss: {total_loss/len(loader):.4f} (Recon: {loss_recon:.2f})"
            )
        
        if self.model.include_diffusion:
            with torch.no_grad():
                learned_alpha = (0.1 + F.softplus(self.model.log_alpha)).cpu().numpy()
            print(f"Learned alpha (inflow scaling): mean={learned_alpha.mean():.4f}, std={learned_alpha.std():.4f}, min={learned_alpha.min():.4f}, max={learned_alpha.max():.4f}")

        self.model.eval()

    def get_regression_coefficients(self, credible_interval: float | None = None) -> pd.DataFrame:
        beta_mu = self.model.beta_mu.detach().cpu().numpy()
        covariate_names = self.design.design_info.column_names
        df = pd.DataFrame(beta_mu, index=covariate_names, columns=self.var_names).melt(ignore_index=False, var_name="Gene", value_name="Mean").reset_index(names="Covariate")
        if credible_interval is not None:
            beta_std = torch.exp(self.model.beta_logstd).detach().cpu().numpy()
            z = stats.norm.ppf(1.0 - (1.0 - credible_interval) / 2.0)
            df["Lower"] = (beta_mu - z * beta_std).flatten(order='F')
            df["Upper"] = (beta_mu + z * beta_std).flatten(order='F')
            df["MinimumCredible"] = np.where(df["Lower"] > 0, df["Lower"], np.where(df["Upper"] < 0, df["Upper"], 0.0))
        return df

    def get_corrected_expression(self, threshold: float = 1e-4, batch_size: int = 4096, n_samples: int = 10) -> csr_matrix:
        self.model.eval()
        from torch.utils.data import DataLoader, TensorDataset
        loader = DataLoader(TensorDataset(torch.arange(self.m), self.data.x), batch_size=batch_size, shuffle=False)
        rows, cols, data = [], [], []
        current_row = 0
        with torch.no_grad():
            for batch_idx, batch_x in loader:
                batch_idx, batch_x = batch_idx.to(self.device), batch_x.to(self.device)
                x_sub_tensor = self.x_dense[batch_idx] if self.x_dense is not None else torch.tensor(self.X[batch_idx.cpu().numpy()].toarray(), dtype=torch.float32, device=self.device)
                if self.model.include_size_factor:
                    log_sf = self.model.log_sf_embed(batch_idx).squeeze(-1)
                    encoder_in = torch.log1p(x_sub_tensor / (torch.exp(log_sf).unsqueeze(-1) + 1e-8) * 1000.0)
                else:
                    log_sf = None
                    encoder_in = torch.log1p(x_sub_tensor)
                mu, logstd = self.model.encoder(encoder_in)
                beta, std = self.model.beta_mu, torch.exp(logstd)
                lam_sum = torch.zeros((batch_idx.size(0), self.n), device=self.device)
                for _ in range(n_samples):
                    z = mu + torch.randn_like(std) * std
                    out = self.model(
                        encoder_in,
                        batch_x,
                        pi_prior=None, # We want corrected (lam), so skip diffusion
                        inflow=None,
                        log_size_factor=log_sf
                    )
                    # out[0] is x_hat, which is lam because pi_prior/inflow are None
                    lam_sum += out[0]
                lam_np = (lam_sum / n_samples).cpu().numpy()
                lam_np[lam_np < threshold] = 0.0
                r, c = np.nonzero(lam_np)
                rows.append(r + current_row); cols.append(c); data.append(lam_np[r, c])
                current_row += batch_idx.size(0)
        return csr_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))), shape=(self.m, self.n))

    def get_latent_representation(self, batch_size: int = 4096) -> np.ndarray:
        self.model.eval()
        from torch.utils.data import DataLoader, TensorDataset
        loader = DataLoader(TensorDataset(torch.arange(self.m), self.data.x), batch_size=batch_size, shuffle=False)
        all_mu = []
        with torch.no_grad():
            for batch_idx, batch_x in loader:
                batch_idx, batch_x = batch_idx.to(self.device), batch_x.to(self.device)
                x_sub_tensor = self.x_dense[batch_idx] if self.x_dense is not None else torch.tensor(self.X[batch_idx.cpu().numpy()].toarray(), dtype=torch.float32, device=self.device)
                if self.model.include_size_factor:
                    log_sf = self.model.log_sf_embed(batch_idx).squeeze(-1)
                    encoder_in = torch.log1p(x_sub_tensor / (torch.exp(log_sf).unsqueeze(-1) + 1e-8) * 1000.0)
                else:
                    encoder_in = torch.log1p(x_sub_tensor)
                mu, _ = self.model.encoder(encoder_in)
                all_mu.append(mu.cpu().numpy())
        return np.concatenate(all_mu, axis=0)
