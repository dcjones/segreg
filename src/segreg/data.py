import numpy as np
from anndata import AnnData
from scipy.sparse import csr_matrix
from spatialdata import SpatialData


def load_proseg_data(
    data: SpatialData | AnnData,
) -> tuple[AnnData, csr_matrix, csr_matrix, csr_matrix]:
    """Extract expression, inflow, and outflow from a proseg AnnData/SpatialData."""
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


def ols_init_beta(
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
