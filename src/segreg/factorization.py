import math

import numpy as np
import torch
import torch.nn.functional as F
from anndata import AnnData
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from spatialdata import SpatialData
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .data import estimate_phi, load_proseg_data
from .nn import SegregFactorizationVAE
from .training import FactorizationTrainingWrapper


def _nndsvd_h_init(X: csr_matrix, k: int, init_ncells: int = 10000) -> np.ndarray:
    """Compute NNDSVD-based initialization for the H (gene programs) matrix.

    Runs truncated SVD, extracts non-negative factors via the NNDSVD procedure,
    and returns log(H_norm) where H_norm is the row-normalized H factor.
    This is the inverse of the softplus activation applied during the forward pass.
    """
    m, n = int(X.shape[0]), int(X.shape[1])  # type: ignore[index]
    if m > init_ncells:
        rng = np.random.default_rng(42)
        indices = rng.choice(m, init_ncells, replace=False)
        indices.sort()
        X_init = X[indices]
    else:
        X_init = X

    U, S, Vt = svds(X_init.astype(np.float64), k=k)  # type: ignore[misc]
    idx = np.argsort(S)[::-1]
    S = S[idx]
    U = U[:, idx]  # type: ignore[index]
    Vt = Vt[idx, :]  # type: ignore[index]
    H = np.zeros((k, n))
    for j in range(k):
        x = U[:, j]
        y = Vt[j, :]
        xp = np.maximum(x, 0)
        xn = np.abs(np.minimum(x, 0))
        yp = np.maximum(y, 0)
        yn = np.abs(np.minimum(y, 0))
        xpn = np.linalg.norm(xp)
        xnn = np.linalg.norm(xn)
        ypn = np.linalg.norm(yp)
        ynn = np.linalg.norm(yn)
        mp = xpn * ypn
        mn = xnn * ynn
        if mp > mn:
            H[j, :] = yp * xpn * np.sqrt(S[j])
        else:
            H[j, :] = yn * xnn * np.sqrt(S[j])

    H = H + 1e-6
    H_norm = H / H.sum(axis=1, keepdims=True)
    return np.log(H_norm).astype(np.float32)


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
        alpha_reg: float = 100.0,
        r_prior_alpha: float = 2.0,
        r_prior_beta: float = 2.0,
        metagene_reg_strength: float = 0.01,
        init_method: str = "nndsvd",
        likelihood: str = "poisson",
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
        self.alpha_reg = alpha_reg
        self.r_prior_alpha = r_prior_alpha
        self.r_prior_beta = r_prior_beta
        self.metagene_reg_strength = metagene_reg_strength
        self.likelihood = likelihood

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

        self.gene_expression = torch.tensor(mean_expr_raw, dtype=torch.float32)

        h_init = None
        if init_method == "nndsvd":
            h_init_np = _nndsvd_h_init(self.X, n_factors)
            h_init = torch.tensor(h_init_np, dtype=torch.float32)

        self.model = SegregFactorizationVAE(
            self.m,
            self.n,
            n_factors,
            rate_offset=rate_offset,
            include_diffusion=include_diffusion,
            include_size_factor=include_size_factor,
            log_mean_expr=log_mean_expr,
            log_sf_prior=torch.tensor(log_size_factors, dtype=torch.float32),
            init_method=init_method,
            h_init=h_init,
        )

    def fit(
        self,
        nepochs: int = 200,
        batch_size: int = 1024,
        lr: float = 1e-2,
        seed: int | None = 42,
        compile: bool = False,
        quiet: bool = False,
        grad_clip: float | None = 1.0,
        lr_schedule: str = "cosine",
        patience: int = 80,
        min_delta: float = 1e-5,
    ):
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)

        dataset = TensorDataset(torch.arange(self.m), self.log_sf_prior_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        n_batches = len(loader)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        r_weight = 1.0 / n_batches

        if lr_schedule == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=nepochs, eta_min=lr * 1e-3
            )
        elif lr_schedule == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=10, min_lr=lr * 1e-3
            )
        elif lr_schedule == "none":
            scheduler = None
        else:
            raise ValueError(
                f"Unknown lr_schedule: {lr_schedule}. "
                f"Must be one of: 'cosine', 'reduce_on_plateau', 'none'."
            )

        best_loss = float("inf")
        no_improvement = 0

        self.model.to(self.device)
        self.model.train()

        training_wrapper = FactorizationTrainingWrapper(
            self.model,
            self.m,
            self.sf_sigma,
            r_weight=r_weight,
            r_prior_alpha=self.r_prior_alpha,
            r_prior_beta=self.r_prior_beta,
            alpha_reg=self.alpha_reg,
            metagene_reg_strength=self.metagene_reg_strength,
            likelihood=self.likelihood,
            gene_expression=None,
        )

        if compile:
            print("Compiling model...")
            training_wrapper = torch.compile(training_wrapper, mode="reduce-overhead")

        use_amp = self.device.type == "cuda"
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        scaler = torch.amp.GradScaler(enabled=use_amp)

        pbar = tqdm(range(nepochs), desc="Training FactorizationModel", disable=quiet)
        for epoch in pbar:
            total_loss = 0.0
            n_batch = 0
            loss_recon = 0.0
            for batch_idx, batch_log_sf_prior in loader:
                optimizer.zero_grad()
                batch_idx = batch_idx.to(self.device)
                batch_log_sf_prior = batch_log_sf_prior.to(self.device)

                x_sub = self.X[batch_idx.cpu().numpy()].toarray().astype(np.float32)
                x_sub_tensor = torch.from_numpy(x_sub).to(self.device)

                if self.model.include_diffusion:
                    assert self.inflow is not None and self.outflow is not None
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
                    loss, loss_recon_val = training_wrapper(
                        batch_idx,
                        batch_log_sf_prior,
                        x_sub_tensor,
                        inflow_sub_tensor,
                        phi_sub_tensor,
                    )

                scaler.scale(loss).backward()
                if grad_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), grad_clip
                    )
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()
                loss_recon = loss_recon_val.item()
                n_batch += 1

            avg_loss = total_loss / n_batch

            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_loss)
            elif scheduler is not None:
                scheduler.step()

            if best_loss - avg_loss > min_delta:
                best_loss = avg_loss
                no_improvement = 0
            else:
                no_improvement += 1

            pbar.set_description(
                f"Loss: {avg_loss:.4f} (Recon: {loss_recon:.2f}) "
                f"Best: {best_loss:.4f} [{no_improvement}/{patience}] "
                f"lr: {optimizer.param_groups[0]['lr']:.2g}"
            )

            if no_improvement >= patience:
                if not quiet:
                    print(
                        f"Early stopping at epoch {epoch + 1}: "
                        f"no improvement for {patience} epochs"
                    )
                break

        self.model.eval()

    def get_factor_loadings(self) -> np.ndarray:
        """Returns non-negative W matrix (after softplus), shape (n_cells, n_factors)."""
        return (
            F.softplus(self.model.W_embed.weight).detach().cpu().numpy()
        )

    def get_factor_programs(self) -> np.ndarray:
        """Returns non-negative H matrix (after softplus), shape (n_factors, n_genes)."""
        return F.softplus(self.model.H).detach().cpu().numpy()

    def get_corrected_expression(
        self, threshold: float = 1e-4, batch_size: int = 4096
    ) -> csr_matrix:
        self.model.eval()

        loader = DataLoader(
            TensorDataset(torch.arange(self.m)),
            batch_size=batch_size,
            shuffle=False,
        )
        rows, cols, data = [], [], []
        current_row = 0
        with torch.no_grad():
            for (batch_idx,) in loader:
                batch_idx = batch_idx.to(self.device)

                if self.model.include_size_factor:
                    log_sf = self.model.log_sf_embed(batch_idx).squeeze(-1)
                else:
                    log_sf = None

                W = F.softplus(self.model.W_embed(batch_idx))
                H = F.softplus(self.model.H)

                log_rate = W @ H + self.model.gene_bias
                if self.model.include_size_factor and log_sf is not None:
                    log_rate = log_rate + log_sf.unsqueeze(-1)
                lam = torch.exp(torch.clamp(log_rate, min=-15.0, max=15.0))
                lam_np = (lam + self.model.rate_offset).cpu().numpy()

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
        """Returns the non-negative cell factor loadings W (after softplus)."""
        return self.get_factor_loadings()
