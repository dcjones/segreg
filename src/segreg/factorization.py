import numpy as np
import torch
from anndata import AnnData
from scipy.sparse import csr_matrix
from spatialdata import SpatialData
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .data import estimate_phi, load_proseg_data
from .nn import SegregFactorizationVAE
from .training import FactorizationTrainingWrapper


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
        alpha_reg: float = 1.0,
    ):
        adata, self.X, self.inflow, self.outflow = load_proseg_data(data, include_diffusion)

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
        self.alpha_reg = alpha_reg

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
            alpha_reg=self.alpha_reg,
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
                    phi_sub = estimate_phi(x_sub, inflow_sub, outflow_sub)

                    inflow_sub_tensor = torch.from_numpy(inflow_sub).to(self.device)
                    phi_sub_tensor = torch.from_numpy(phi_sub).to(self.device)
                else:
                    inflow_sub_tensor = None
                    phi_sub_tensor = None

                with torch.autocast(
                    device_type=self.device.type, dtype=amp_dtype, enabled=use_amp
                ):
                    loss, loss_recon = training_wrapper(
                        batch_idx,
                        batch_log_sf_prior,
                        x_sub_tensor,
                        inflow_sub_tensor,
                        phi_sub_tensor,
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
                z_mu, z_logstd = self.model.encoder(encoder_in)
                std = torch.exp(z_logstd)
                lam_sum = torch.zeros((batch_idx.size(0), self.n), device=self.device)
                for _ in range(n_samples):
                    z = z_mu + torch.randn_like(std) * std
                    z_offset = self.model.node_decoder(z)
                    W = self.model.W_embed(batch_idx)
                    log_rate = z_offset + W @ self.model.H + self.model.gene_bias
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
            TensorDataset(torch.arange(self.m), self.log_sf_prior_t),
            batch_size=batch_size,
            shuffle=False,
        )
        all_z_mu = []
        with torch.no_grad():
            for batch_idx, _ in loader:
                batch_idx = batch_idx.to(self.device)
                x_sub_tensor = torch.tensor(
                    self.X[batch_idx.cpu().numpy()].toarray(),
                    dtype=torch.float32,
                    device=self.device,
                )
                encoder_in, _ = self.model.prepare_encoder_input(x_sub_tensor, batch_idx)
                z_mu, _ = self.model.encoder(encoder_in)
                all_z_mu.append(z_mu.cpu().numpy())
        return np.concatenate(all_z_mu, axis=0)
