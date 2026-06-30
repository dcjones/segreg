from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch
from anndata import AnnData
from scipy.sparse import csr_matrix
from scipy.sparse import csr_matrix as make_csr
from spatialdata import SpatialData


def load_proseg_data(
    data: SpatialData | AnnData,
    include_diffusion: bool = True,
) -> tuple[
    AnnData, csr_matrix, csr_matrix | None, csr_matrix | None, csr_matrix | None
]:
    """Extract expression, and optionally inflow/outflow, from an AnnData/SpatialData.

    When include_diffusion is True, requires proseg data and reads inflow/outflow.
    When False, skips the proseg check and returns None for inflow/outflow, allowing
    general AnnData or SpatialData files to be used.
    """
    if isinstance(data, AnnData):
        adata = data
    elif isinstance(data, SpatialData):
        adata = data.tables["table"]
    else:
        raise ValueError("data must be an AnnData or SpatialData object")

    X = adata.X.tocsr() if not isinstance(adata.X, csr_matrix) else adata.X

    if not include_diffusion:
        return adata, X, None, None, None

    if "proseg_run" not in adata.uns:
        raise ValueError(
            "This is not a proseg spatialdata file. "
            "Set include_diffusion=False to use general AnnData/SpatialData."
        )

    inflow = adata.layers["expected_inflow"].tocsr()
    assert isinstance(inflow, csr_matrix)
    outflow = adata.layers["expected_outflow"].tocsr()
    assert isinstance(outflow, csr_matrix)

    # expected counts
    T = X + outflow - inflow

    # expected proportion lost to outflow
    T_recip = T.copy()
    nz_mask = T_recip.data != 0
    T_recip.data[nz_mask] = 1.0 / T_recip.data[nz_mask]
    φ = outflow.multiply(T_recip)

    return adata, X, inflow, outflow, φ


def estimate_phi(X: np.ndarray, inflow: np.ndarray, outflow: np.ndarray) -> np.ndarray:
    """Estimate phi_cg, the fraction of a cell's true transcripts lost to neighbors.

    T_cg = X_cg + outflow_cg - inflow_cg is the true-count identity from the paper,
    and phi_cg = outflow_cg / T_cg (taking phi_cg = 0 where T_cg <= 0, which also
    guards against estimation noise pushing T below zero).
    """
    T = X + outflow - inflow
    phi = np.zeros_like(outflow)
    mask = T > 0
    phi[mask] = outflow[mask] / T[mask]
    return np.clip(phi, 0.0, 1.0)


def ols_init_beta(
    X: csr_matrix,
    inflow: csr_matrix | None,
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
    # potential and should be shrunk more aggressively. Only applies with proseg data.
    if inflow is not None:
        avg_obs = np.asarray(X.mean(axis=0)).squeeze()
        avg_inflow = np.asarray(inflow.mean(axis=0)).squeeze()
        suspect_score = avg_inflow / (avg_obs + 1e-8)
        suspect_shrinkage = 1.0 / (1.0 + 50.0 * suspect_score)
    else:
        suspect_shrinkage = np.ones(n, dtype=np.float32)

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


@dataclass
class CSRMatrixBatch:
    data: torch.Tensor
    indices: torch.Tensor
    indptr: torch.Tensor
    row_idx: torch.Tensor
    batch_m: int

    def __init__(self, M: csr_matrix, batch_idx: npt.NDArray[np.int64], use_pin: bool):
        sliced = M[batch_idx, :]
        nnz_per_row = np.diff(sliced.indptr)
        row_idx_np = np.repeat(np.arange(sliced.shape[0], dtype=np.int64), nnz_per_row)

        def _maybe_pin(t: torch.Tensor) -> torch.Tensor:
            return t.pin_memory() if use_pin else t

        self.data = _maybe_pin(torch.from_numpy(sliced.data.copy()))
        self.indices = _maybe_pin(torch.from_numpy(sliced.indices.astype(np.int64)))
        self.indptr = _maybe_pin(torch.from_numpy(sliced.indptr.astype(np.int64)))
        self.row_idx = _maybe_pin(torch.from_numpy(row_idx_np))
        self.batch_m = sliced.shape[0]

    def to_csr_tensor(
        self, n: int, non_blocking: bool, device: torch.device
    ) -> torch.Tensor:
        return torch.sparse_csr_tensor(
            crow_indices=self.indptr.to(device, non_blocking=non_blocking),
            col_indices=self.indices.to(device, non_blocking=non_blocking),
            values=self.data.to(device, non_blocking=non_blocking),
            size=(self.batch_m, n),
            dtype=torch.float32,
            device=device,
        )


# TODO: OK, this was copied from countdown, but now we need to jointly sample three sparse matrices:
# X, inflow, and phi
#
# These do not necessarily have the same
class SparseBatchSampler:
    """
    Samples batches of rows from a CSR matrix as torch sparse_csr_tensor objects.

    Precomputes all batches at initialization as pinned CPU tensors (including
    row indices for the loss function), then transfers to GPU during iteration
    using non-blocking transfers for better CPU/GPU overlap.
    """

    def __init__(
        self,
        X: csr_matrix,
        inflow: csr_matrix,
        φ: csr_matrix,
        batch_size: int,
        device: torch.device,
    ):
        m, n = X.shape
        assert inflow.shape == φ.shape == (m, n)
        self.X = X.astype(np.float32)
        self.inflow = inflow.astype(np.float32)
        self.φ = φ.astype(np.float32)
        self.m = m
        self.n = n
        self.batch_size = batch_size
        self.device = device
        self.use_pin = device.type == "cuda"
        self._batches: list[tuple[CSRMatrixBatch, CSRMatrixBatch, CSRMatrixBatch]] = []
        self.shuffle()

    def shuffle(self) -> None:
        """Rebuild the batched pre-sliced CPU tensors with a fresh row permutation.

        Called once at construction and once per epoch by the training loop so
        that batch order/composition varies across epochs, mirroring
        DenseRowSampler. Keeping the per-batch CSR slices precomputed on the
        CPU (pinned when on CUDA) preserves the cheap non-blocking transfer
        path while still reshuffling between epochs.
        """

        idx = np.arange(self.m)
        np.random.shuffle(idx)

        self._batches = []
        for fr in range(0, self.m, self.batch_size):
            to = min(fr + self.batch_size, self.m)
            batch_idx = idx[fr:to].copy()
            batch_idx.sort()

            X_batch = CSRMatrixBatch(self.X, batch_idx, self.use_pin)
            inflow_batch = CSRMatrixBatch(self.inflow, batch_idx, self.use_pin)
            φ_batch = CSRMatrixBatch(self.φ, batch_idx, self.use_pin)
            self._batches.append((X_batch, inflow_batch, φ_batch))

    def __iter__(self):
        nb = self.device.type == "cuda"
        with torch.sparse.check_sparse_tensor_invariants(enable=False):
            for X_batch, inflow_batch, φ_batch in self._batches:
                X_batch_csr = X_batch.to_csr_tensor(self.n, nb, self.device)
                inflow_batch_csr = inflow_batch.to_csr_tensor(self.n, nb, self.device)
                φ_batch_csr = φ_batch.to_csr_tensor(self.n, nb, self.device)

                yield (
                    X_batch_csr,
                    inflow_batch_csr,
                    φ_batch_csr,
                    X_batch.row_idx.to(self.device, non_blocking=nb),
                )
