import math
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from anndata import AnnData
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from spatialdata import SpatialData
from tqdm import tqdm

from .data import SparseBatchSampler, estimate_phi, load_proseg_data
from .training import FactorizationTrainingWrapper

warnings.filterwarnings(
    "ignore",
    message="Sparse CSR tensor support is in beta state",
    category=UserWarning,
)


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


class SparseLinear(nn.Module):
    """
    Linear layer that handles both sparse CSR and dense input.
    Weight is stored as [in_features, out_features] for torch.sparse.mm compatibility.
    Initialized with LeCun normal (std = sqrt(1/fan_in)).
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        std = math.sqrt(1.0 / in_features)
        nn.init.normal_(self.weight, 0.0, std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.layout == torch.sparse_csr:
            return torch.sparse.mm(x, self.weight) + self.bias
        return x @ self.weight + self.bias


class SparseFactorizationEncoder(nn.Module):
    """
    Very simple encoder that maps a sparse count matrix into metagene rates.
    """

    def __init__(self, n: int, k: int):
        super().__init__()
        self.layer = SparseLinear(n, k)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.layer(X))


class FactorizationVAE(nn.Module):
    def __init__(
        self, n: int, k: int, v_init: np.ndarray | None, include_diffusion: bool
    ):
        super().__init__()

        if v_init is None:
            v = torch.empty(k, n)
            nn.init.normal_(v, std=1.0 / math.sqrt(n))
            self.v = nn.Parameter(v)
        else:
            self.v = nn.Parameter(torch.from_numpy(v_init))

        self.encoder = SparseFactorizationEncoder(n, k)
        self.include_diffusion = include_diffusion
        if include_diffusion:
            self.log_α = nn.Parameter(torch.full((n,), -3.0))

    def v_norm(self) -> torch.Tensor:
        return F.softmax(self.v, dim=1)

    def forward(
        self, X: torch.Tensor, inflow: torch.Tensor | None, φ: torch.Tensor | None
    ):
        u = self.encoder(X)
        λ = u @ self.v_norm()

        if self.include_diffusion:
            assert inflow is not None and φ is not None
            α = torch.sigmoid(self.log_α)
            retension = torch.exp(-α * φ.to_dense())
            δ = α * inflow.to_dense()
            λ = retension * λ + δ

        return u, λ

    def metagene_regularization(self):
        pass


def _sparse_row_col_indices(
    X: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (row_idx, col_idx) arrays indexing every non-zero element in a sparse_csr_tensor."""
    crow = X.crow_indices()
    col_idx = X.col_indices()
    batch_size = X.shape[0]
    row_idx = torch.repeat_interleave(
        torch.arange(batch_size, device=X.device),
        crow[1:] - crow[:-1],
    )
    return row_idx, col_idx


def loss_fn(
    model: FactorizationVAE,
    X: torch.Tensor,
    inflow: torch.Tensor | None = None,
    φ: torch.Tensor | None = None,
    row_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    u, λ = model(X, inflow, φ)
    v = model.v_norm()

    loss = -poisson_logprob_sparse(λ, X, constant_terms=False, row_idx=row_idx)
    if model.include_diffusion:
        log_α = model.log_α if model.include_diffusion else None
        # TODO: Prior on α. Maybe just Normal(0, σ) neg log prob.

    # TODO: metagene regularization on v. I think we want to do min-volume regularization here.

    return loss


def poisson_logprob_sparse(
    λ: torch.Tensor,
    X: torch.Tensor,
    constant_terms: bool = False,
    row_idx: torch.Tensor | None = None,
):
    """Log probability for sparse CSR input under Poisson likelihood."""
    col_idx = X.col_indices()
    if row_idx is None:
        row_idx, col_idx = _sparse_row_col_indices(X)
    x_data = X.values()
    lp = (x_data * torch.log(λ[row_idx, col_idx].clamp(1e-8))).sum() - λ.sum()
    if constant_terms:
        lp -= torch.lgamma(x_data + 1).sum()

    return lp


class FactorizationModel:
    X: csr_matrix
    inflow: csr_matrix | None
    φ: csr_matrix | None
    device: torch.device
    model: FactorizationVAE
    m: int
    n: int

    def __init__(
        self,
        data: SpatialData | AnnData,
        n_factors: int,
        batch_size: int | None = 4096,
        include_diffusion: bool = True,
        sf_sigma: float = 0.5,
        rate_offset: float = 1e-2,
        alpha_reg: float = 100.0,
        r_prior_alpha: float = 2.0,
        r_prior_beta: float = 2.0,
        metagene_reg_type: str = "none",
        metagene_reg_strength: float = 0.01,
        init_method: str = "nndsvd",
        likelihood: str = "poisson",
    ):
        adata, self.X, self.inflow, outflow, self.φ = load_proseg_data(
            data, include_diffusion
        )

        self.m, self.n = adata.shape
        self.n_factors = n_factors
        self.var_names = adata.var_names
        self.obs_names = adata.obs_names

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.sf_sigma = sf_sigma
        self.alpha_reg = alpha_reg
        self.r_prior_alpha = r_prior_alpha
        self.r_prior_beta = r_prior_beta
        self.metagene_reg_strength = metagene_reg_strength
        self.likelihood = likelihood

        mean_expr_raw = np.asarray(self.X.mean(axis=0)).squeeze().astype(np.float32)
        log_mean_expr = torch.tensor(np.log(mean_expr_raw + 1e-4), dtype=torch.float32)

        self.gene_expression = torch.tensor(mean_expr_raw, dtype=torch.float32)

        v_init = None
        if init_method == "nndsvd":
            v_init = _nndsvd_h_init(self.X, n_factors)

        self.model = FactorizationVAE(
            self.n,
            n_factors,
            v_init,
            include_diffusion=include_diffusion,
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

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

        if compile:
            print("Compiling model...")
            train_loss_fn = torch.compile(loss_fn)
        else:
            train_loss_fn = loss_fn

        best_loss = float("inf")
        no_improvement_count = 0

        # TODO: Need to handle the case where inflow and φ are None
        # if we want this to be more widely applicable. (E.g. if we want to replace countdown
        # entirely with this code.)
        batch_sampler = SparseBatchSampler(
            self.X, self.inflow, self.φ, batch_size, device
        )

        self.model.to(self.device)
        self.model.train()

        with tqdm(range(nepochs), desc="Training", unit="epoch", disable=quiet) as pbar:
            for epoch in pbar:
                batch_sampler.shuffle()
                epoch_loss_sum = 0.0
                epoch_batch_count = 0

                for (
                    X_batch,
                    inflow_batch,
                    φ_batch,
                    precomputed_row_idx,
                ) in batch_sampler:
                    optimizer.zero_grad(set_to_none=True)
                    loss = train_loss_fn(
                        self.model,
                        X_batch,
                        inflow_batch,
                        φ_batch,
                        row_idx=precomputed_row_idx,
                    )
                    loss.backward()
                    if grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), grad_clip
                        )
                    optimizer.step()

                    epoch_loss_sum += loss.detach().item()
                    epoch_batch_count += 1
                    pass

                if not np.isfinite(epoch_loss_sum):
                    raise ValueError(f"Non-finite loss: {epoch_loss_sum}")

                # Step LR scheduler once per epoch.
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(epoch_loss_sum)
                elif scheduler is not None:
                    scheduler.step()

                if best_loss - epoch_loss_sum > min_delta:
                    best_loss = epoch_loss_sum
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                pbar.set_postfix(
                    loss=f"{epoch_loss_sum:.4f}",
                    best=f"{best_loss:.4f}",
                    patience=f"{no_improvement_count}/{patience}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2g}",
                )

                if no_improvement_count >= patience:
                    pbar.write(
                        f"Early stopping at epoch {epoch + 1}: no improvement for {patience} epochs"
                    )
                    break

    def get_factor_loadings(self, batch_size: int = 4096) -> np.ndarray:
        """Returns non-negative W matrix (after softplus), shape (n_cells, n_factors)."""

        Xnmf = np.zeros((self.m, self.n_factors), dtype=np.float32)
        self.model.eval()
        with torch.no_grad():
            for start_idx in range(0, self.m, batch_size):
                end_idx = min(start_idx + batch_size, self.m)

                X_chunk = self.X[start_idx:end_idx, :]
                X_chunk_tensor = torch.sparse_csr_tensor(
                    X_chunk.indptr,
                    X_chunk.indices,
                    X_chunk.data,
                    size=(end_idx - start_idx, self.n),
                    dtype=torch.float32,
                    device=self.device,
                ).to(self.device)

                encoded_chunk = self.model.encoder(X_chunk_tensor)
                Xnmf[start_idx:end_idx, :] = encoded_chunk.cpu().numpy()

        return Xnmf

    def get_factor_programs(self) -> np.ndarray:
        """Returns non-negative H matrix (after softplus), shape (n_factors, n_genes)."""
        return self.model.v_norm().detach().cpu().numpy().T

    def get_corrected_expression(
        self, threshold: float = 1e-4, batch_size: int = 4096
    ) -> csr_matrix:
        self.model.eval()

        # TODO: Gotta rewrite this. Though I'm not sure I even need it.

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
