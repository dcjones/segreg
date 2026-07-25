from typing import cast

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
from anndata import AnnData
from patsy import dmatrix
from patsy.design_info import DesignMatrix
from scipy.sparse import csr_matrix
from spatialdata import SpatialData
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import Data
from tqdm import tqdm

from .data import (
    clip_phi,
    load_inflow_var,
    load_proseg_data,
    ols_init_beta,
    slice_csr_to_sparse_tensor,
)
from .nn import SegregVAE
from .training import SegregTrainingWrapper


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
        # Retention (multiplicative outflow correction exp(-alpha*phi)) is off by
        # default: A/B testing across 5 cell types -- with and without a decoupled
        # retention alpha -- showed it is inert for decontamination (retention-only
        # tracks no-diffusion) and does not recover own-signal attenuation either,
        # while the additive inflow term (delta) carries the entire correction.
        include_retention: bool = False,
        include_delta: bool = True,
        separate_retention_alpha: bool = False,
        stochastic_alpha: bool = False,
        component_alpha: bool = False,
        likelihood: str = "nb_mean",
        contam_var: str = "poisson",
        retention_form: str | None = None,
        conv_max: int | None = None,
        include_size_factor: bool = True,
        sf_sigma: float = 0.5,
        rate_offset: float = 1e-2,
        hidden_channels: int = 128,
        latent_dim: int = 64,
        kappa: float = 1000.0,
        beta_prior_scale: float = 1.0,
        interaction_prior_scale: float | None = None,
        interaction_suspect_shrinkage: bool = True,
        interaction_columns: list[str] | None = None,
        alpha_reg: float = 1.0,
    ):
        adata, self.X, self.inflow, self.outflow, phi = load_proseg_data(
            data, include_diffusion
        )

        # Posterior variance of the inflow, needed only by the non-Poisson
        # contamination arms (see load_inflow_var / the family notes in losses.py).
        if include_diffusion and contam_var != "poisson":
            self.inflow_var = load_inflow_var(adata)
        else:
            self.inflow_var = None

        self.m, self.n = adata.shape
        self.var_names = adata.var_names
        self.obs_names = adata.obs_names

        # conv_max only needs to cover the Poisson(delta) tail (delta = inflow at
        # alpha=1), not the max count -- the j <= x constraint is masked in. Size it
        # from the data's max inflow so a whole-transcriptome panel (max inflow ~64)
        # gets an exact convolution without the caller tuning it; entries below it
        # are exact regardless. Floored so tiny datasets still get a sane window.
        if likelihood in ("nb_conv", "nb_mm") and conv_max is None:
            if self.inflow is not None and self.inflow.nnz > 0:
                imax = float(self.inflow.data.max())
                conv_max = int(np.ceil(imax + 8.0 * np.sqrt(imax)))
            else:
                conv_max = 64
            conv_max = int(np.clip(conv_max, 32, 512))
        elif conv_max is None:
            conv_max = 64
        self.conv_max = conv_max

        # Per-cell proseg component (its point-estimate mixture assignment) for
        # component_alpha. Factorized to contiguous 0..C-1 codes.
        self.component_alpha = component_alpha
        if component_alpha:
            if "component" not in adata.obs.columns:
                raise ValueError(
                    "component_alpha=True requires a 'component' column in adata.obs"
                )
            comp_codes = pd.factorize(np.asarray(adata.obs["component"]))[0]
            self.component = torch.tensor(comp_codes, dtype=torch.long)
            self.n_components = int(comp_codes.max()) + 1
        else:
            self.component = None
            self.n_components = 1

        # phi_cg is a fixed function of the (fixed) counts/inflow/outflow, so use
        # the sparse phi load_proseg_data already computed rather than re-deriving
        # it per batch. clip_phi applies estimate_phi's [0,1] / T<=0 semantics to
        # the raw outflow/T values. Kept sparse; sliced per batch during training.
        self.phi = clip_phi(phi) if include_diffusion else None

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
        self.alpha_reg = alpha_reg
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

        beta_init_np, beta_prior_scale_matrix, beta_logstd_init = ols_init_beta(
            self.X,
            self.inflow,
            design_np,
            sf_col,
            log_mean_expr.numpy(),
            covariate_names,
            self.m,
            self.n,
            beta_prior_scale,
            interaction_prior_scale=interaction_prior_scale,
            interaction_suspect_shrinkage=interaction_suspect_shrinkage,
            interaction_columns=interaction_columns,
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
            include_retention=include_retention,
            include_delta=include_delta,
            separate_retention_alpha=separate_retention_alpha,
            stochastic_alpha=stochastic_alpha,
            component_alpha=component_alpha,
            n_components=self.n_components,
            likelihood=likelihood,
            contam_var=contam_var,
            retention_form=retention_form,
            conv_max=self.conv_max,
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

        use_pin = self.device.type == "cuda"
        nb = use_pin  # non-blocking transfers only help alongside pinned memory

        dataset = TensorDataset(
            torch.arange(self.m), self.data.x, self.data.log_sf_prior
        )
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, pin_memory=use_pin
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.to(self.device)
        self.model.train()

        training_wrapper = SegregTrainingWrapper(
            self.model,
            self.beta_prior_scale_t,
            self.m,
            self.sf_sigma,
            alpha_reg=self.alpha_reg,
        )

        if compile:
            # torch.compile provides ~20% speedup on CUDA.
            # Requires CUDA toolkit (ptxas) to be in PATH and TRITON_PTXAS_PATH set if not standard.
            print("Compiling model...")
            training_wrapper = torch.compile(training_wrapper, mode="reduce-overhead")

        use_amp = self.device.type == "cuda"
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        # GradScaler is only needed for fp16; bf16 has the dynamic range to train
        # without loss scaling. Skipping it for bf16 also removes a per-step
        # inf-check that reads a GPU flag back to the CPU (a sync), which would
        # otherwise serialize CPU batch-prep against GPU compute.
        use_scaler = use_amp and amp_dtype == torch.float16
        scaler = torch.amp.GradScaler(enabled=use_scaler)

        pbar = tqdm(range(nepochs), desc="Training Segreg", disable=quiet)
        for epoch in pbar:
            if kl_annealing and nepochs > 1:
                progress = min(1.0, (epoch + 1) / (nepochs // 2))
                current_beta_kl = beta_kl * progress
            else:
                current_beta_kl = beta_kl

            current_beta_kl_t = torch.tensor(current_beta_kl, device=self.device)

            # Accumulate on-device and sync once per epoch (below) rather than
            # calling loss.item() every batch, so the CPU can run ahead and prepare
            # the next batch while the GPU is still computing this one.
            total_loss = torch.zeros((), device=self.device)
            last_recon = torch.zeros((), device=self.device)
            for batch_idx, batch_x, batch_log_sf_prior in loader:
                optimizer.zero_grad(set_to_none=True)
                idx_np = batch_idx.numpy()
                batch_idx = batch_idx.to(self.device, non_blocking=nb)
                batch_x = batch_x.to(self.device, non_blocking=nb)
                batch_log_sf_prior = batch_log_sf_prior.to(self.device, non_blocking=nb)

                x_sub_tensor, row_idx = slice_csr_to_sparse_tensor(
                    self.X, idx_np, self.n, self.device, use_pin, nb
                )

                # Only slice the batch for terms the model actually uses, so an
                # inflow-only or outflow-only ablation skips the unneeded transfer.
                if self.model.include_diffusion and self.model.include_delta:
                    assert self.inflow is not None
                    inflow_sub_tensor, _ = slice_csr_to_sparse_tensor(
                        self.inflow, idx_np, self.n, self.device, use_pin, nb
                    )
                else:
                    inflow_sub_tensor = None

                if inflow_sub_tensor is not None and self.inflow_var is not None:
                    inflow_var_sub_tensor, _ = slice_csr_to_sparse_tensor(
                        self.inflow_var, idx_np, self.n, self.device, use_pin, nb
                    )
                else:
                    inflow_var_sub_tensor = None

                if self.model.include_diffusion and self.model.include_retention:
                    assert self.phi is not None
                    phi_sub_tensor, _ = slice_csr_to_sparse_tensor(
                        self.phi, idx_np, self.n, self.device, use_pin, nb
                    )
                else:
                    phi_sub_tensor = None

                if self.component_alpha:
                    assert self.component is not None
                    batch_component = self.component[batch_idx.cpu()].to(
                        self.device, non_blocking=nb
                    )
                else:
                    batch_component = None

                with torch.autocast(
                    device_type=self.device.type, dtype=amp_dtype, enabled=use_amp
                ):
                    loss, loss_recon = training_wrapper(
                        batch_idx,
                        batch_x,
                        batch_log_sf_prior,
                        x_sub_tensor,
                        inflow_sub_tensor,
                        phi_sub_tensor,
                        row_idx,
                        current_beta_kl_t,
                        batch_component,
                        inflow_var_sub_tensor,
                    )

                if use_scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                total_loss += loss.detach()
                last_recon = loss_recon.detach()

            pbar.set_description(
                f"Loss: {(total_loss / len(loader)).item():.4f} "
                f"(Recon: {last_recon.item():.2f})"
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

        loader = DataLoader(
            TensorDataset(torch.arange(self.m), self.data.x),
            batch_size=batch_size,
            shuffle=False,
        )
        rows, cols, data = [], [], []
        current_row = 0
        with torch.no_grad():
            for batch_idx, batch_x in loader:
                idx_np = batch_idx.numpy()
                batch_idx, batch_x = batch_idx.to(self.device), batch_x.to(self.device)
                x_sub_tensor, _ = slice_csr_to_sparse_tensor(
                    self.X, idx_np, self.n, self.device, use_pin=False, non_blocking=False
                )
                encoder_in, log_sf = self.model.prepare_encoder_input(
                    x_sub_tensor, batch_idx
                )
                z_mu, z_logstd = self.model.encoder(encoder_in)
                std = torch.exp(z_logstd)
                lam_sum = torch.zeros((batch_idx.size(0), self.n), device=self.device)
                for _ in range(n_samples):
                    z = z_mu + torch.randn_like(std) * std
                    z_offset = self.model.node_decoder(z)
                    log_rate = z_offset + batch_x @ self.model.beta_mu + self.model.gene_bias
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

        loader = DataLoader(
            TensorDataset(torch.arange(self.m), self.data.x),
            batch_size=batch_size,
            shuffle=False,
        )
        all_z_mu = []
        with torch.no_grad():
            for batch_idx, batch_x in loader:
                idx_np = batch_idx.numpy()
                batch_idx, batch_x = batch_idx.to(self.device), batch_x.to(self.device)
                x_sub_tensor, _ = slice_csr_to_sparse_tensor(
                    self.X, idx_np, self.n, self.device, use_pin=False, non_blocking=False
                )
                encoder_in, _ = self.model.prepare_encoder_input(
                    x_sub_tensor, batch_idx
                )
                z_mu, _ = self.model.encoder(encoder_in)
                all_z_mu.append(z_mu.cpu().numpy())
        return np.concatenate(all_z_mu, axis=0)
