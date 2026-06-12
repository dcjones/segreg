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

from .data import load_proseg_data, ols_init_beta
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
        include_size_factor: bool = True,
        sf_sigma: float = 0.5,
        rate_offset: float = 1e-2,
        hidden_channels: int = 128,
        latent_dim: int = 64,
        kappa: float = 1000.0,
        beta_prior_scale: float = 1.0,
        inflow_scale_reg: float = 1.0,
    ):
        adata, self.X, self.inflow, self.outflow = load_proseg_data(
            data, include_diffusion
        )

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
                batch_idx, batch_x = batch_idx.to(self.device), batch_x.to(self.device)
                x_sub_tensor = torch.tensor(
                    self.X[batch_idx.cpu().numpy()].toarray(),
                    dtype=torch.float32,
                    device=self.device,
                )
                encoder_in, log_sf = self.model.prepare_encoder_input(
                    x_sub_tensor, batch_idx
                )
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
                encoder_in, _ = self.model.prepare_encoder_input(
                    x_sub_tensor, batch_idx
                )
                mu, _ = self.model.encoder(encoder_in)
                all_mu.append(mu.cpu().numpy())
        return np.concatenate(all_mu, axis=0)
