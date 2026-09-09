"""Build the drift covariate `u` from the segmenter's own output.

`u[k, g]` is the direction along which missegmentation biases design column `k`'s
coefficient for gene `g`. `Regression(include_drift=True)` adds `κ*u` to the
coefficient and reports `β`; this module is where `u` comes from when the caller
does not supply one.

WHY IT LIVES HERE NOW. It used to be preprocessing -- a script that clustered the
counts, built the covariate and wrote a table the caller passed in as `drift=`.
Every input it needs is already on the `Regression` object (the counts, the
segmenter's transition matrix, the design matrix), and requiring a separate build
step meant the leading correction could not be run from a single call. The external
`drift=` seam is kept: an oracle-built or otherwise perturbed `u` is how the
covariate's own robustness gets measured, and that must stay expressible.

--- the construction, in six steps ---------------------------------------------

None of them sees truth, a reference, or an annotation:

  1. Cluster the segmented counts (Leiden; normalize -> log1p -> HVG -> scale ->
     PCA -> kNN). The clustering supplies POOLING, not labels -- an annotation
     would do as well, and per-cell rates with NO pooling collapse on a
     whole-transcriptome panel, where one cell's expression vector is too noisy to
     say what a neighbourhood expresses.
  2. P[c, g] -- the mean count of g over cluster c, rows normalized to sum to 1, so
     it is a COMPOSITION: gene g's share of a cell's counts.
  3. M[i, c] -- cell i's incoming mixing mass from cluster c, excluding its own
     diagonal; sw[i] -- that diagonal. Normalized so M[i].sum() + sw[i] == 1.
     BACKGROUND IS EXCLUDED: u describes cell-to-cell mixing only.
  4. F[i, g] = (M[i] @ P)[g] and O[i, g] = sw[i] * P[c_i, g] -- predicted donor and
     self rates.
  5. frac[i, g] = F / (F + O) -- the PREDICTED foreign share of the observed
     expression. Never materialized in full (3 GB on a WTA panel); built in gene
     chunks.
  6. u[:, g] = (X'WX)^-1 X'W frac[:, g] -- the reduction.

--- why the projection, and not a within-type slope ----------------------------

The perturbation missegmentation induces on a coefficient vector is the WEIGHTED
projection of the per-cell log-rate perturbation onto the design: standard
omitted-variable bias for a GLM, where the weights come from expanding the score
equation (`e ≈ κ·frac`, so the bias is `κ·u`).

The original form took a univariate slope of `frac` on an exposure covariate within
each cell type. That is valid ONLY for a design giving each type's exposure column
disjoint support alongside a full set of per-type intercepts -- with the intercepts
absorbing the mean, a within-group centred slope IS the projection coefficient. Any
richer design (overlapping neighbourhood features, continuous covariates, no
cell-type partition at all) breaks it SILENTLY, attributing shared variation to
whichever column the slope happened to be taken against. Projecting onto the actual
design matrix cannot make that mistake, and it needs no grouping variable, which is
also what lets this live inside the model: the design is right here.

--- the weights (`drift_weights`) ----------------------------------------------

W = diag((dμ/dη)² / V(μ)), the GLM's own IRLS weight, in three tiers:

  none  W = I. Ordinary projection, gene-independent, cheapest. THE DEFAULT, and
        the tier that provably reproduces the within-type slope.
  size  W = diag(observed total counts). For a log link the offset dominates the
        per-cell variation, so relative weights are ~exp(log_size_i) for every gene.
  irls  Adds the NB saturation w = μr/(r+μ). Note that ANY separable mean
        μ_ig = counts_i * p_g gives the same projection as `size` -- the per-gene
        scalar cancels out of (X'WX)^-1 X'W -- so only the saturation makes this
        tier differ at all, and only for genes with μ comparable to r. It is
        therefore computed exactly for just those genes and falls back to `size`
        elsewhere.

MEASURED, the tier does not matter: three tiers span 0.022 (313-gene panel) /
0.013 (16.7k-gene panel) of median R², and the ranking INVERTS between the two
datasets. Which is why the default is the simplest one.
"""

import warnings

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.sparse as sp

EPS = 1e-9


def leiden_clusters(X, var_names, resolution: float = 1.0, n_pcs: int = 50,
                    n_top_genes: int = 2000, seed: int = 0) -> npt.NDArray[np.str_]:
    """Leiden on the segmented counts. Nothing here sees truth or a reference.

    A COPY is clustered, so the caller's count matrix is untouched.
    """
    try:
        import scanpy as sc
    except ImportError as e:  # pragma: no cover - environment problem, not logic
        raise ImportError(
            "computing the drift covariate needs scanpy (and igraph) for the "
            "Leiden step. Install them, pass `drift_clusters=` with labels you "
            "already have, or pass a prebuilt `drift=` table."
        ) from e
    from anndata import AnnData

    b = AnnData(X=X.copy())
    b.var_names = np.asarray(var_names).astype(str)
    sc.pp.normalize_total(b, target_sum=1e4)
    sc.pp.log1p(b)
    if n_top_genes and b.n_vars > n_top_genes:
        sc.pp.highly_variable_genes(b, n_top_genes=n_top_genes)
        b = b[:, b.var["highly_variable"]].copy()
    sc.pp.scale(b, max_value=10)
    sc.tl.pca(b, n_comps=min(n_pcs, b.n_vars - 1, b.n_obs - 1), random_state=seed)
    sc.pp.neighbors(b, n_neighbors=15, random_state=seed)
    sc.tl.leiden(b, resolution=resolution, random_state=seed,
                 flavor="igraph", n_iterations=2, directed=False)
    return np.asarray(b.obs["leiden"]).astype(str)


def moment_dispersion(X, floor: float = 0.01, cap: float = 1e6):
    """Per-gene NB dispersion r by moments, from the counts alone.

    Var = μ + μ²/r  =>  r = μ²/(Var − μ). Only used to LOCATE the genes whose IRLS
    weight actually saturates; a crude r is enough for that, and no fit is needed,
    which is what keeps the `irls` tier computable before the fit.
    """
    n = X.shape[0]
    mu = np.asarray(X.mean(axis=0)).ravel()
    m2 = (np.asarray(X.multiply(X).mean(axis=0)).ravel() if sp.issparse(X)
          else (np.asarray(X) ** 2).mean(axis=0))
    var = np.maximum(m2 - mu ** 2, 0.0) * n / max(n - 1, 1)
    over = var - mu
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(over > 0, mu ** 2 / np.maximum(over, 1e-12), cap)
    return np.clip(np.nan_to_num(r, nan=cap, posinf=cap), floor, cap)


def project(Xd, frac, w=None):
    """(X'WX)^-1 X'W frac, for a shared (gene-independent) weight vector."""
    Wx = Xd if w is None else Xd * w[:, None]
    XtWX = Xd.T @ Wx
    XtWf = Wx.T @ frac
    return np.linalg.solve(XtWX + 1e-8 * np.eye(Xd.shape[1]), XtWf)


def cluster_profiles(X, code: npt.NDArray[np.integer], n_clusters: int):
    """P[c, g]: cluster c's mean count of g, rows normalized to a composition."""
    P = np.zeros((n_clusters, X.shape[1]))
    for c in range(n_clusters):
        sel = np.flatnonzero(code == c)
        if not len(sel):
            continue
        P[c] = np.asarray(X[sel].sum(axis=0)).ravel() / len(sel)
    return P / np.maximum(P.sum(axis=1, keepdims=True), EPS)


def donor_mixture(A, code: npt.NDArray[np.integer], n_clusters: int):
    """M[i, c] and sw[i] from the segmenter's RAW transition matrix.

    `A` is the raw `obsp["state_transitions"]`, not the row-normalized mixing matrix
    the likelihood uses: the normalization here excludes background, so it divides
    by the cell-to-cell mass plus the diagonal rather than by the row total. It is
    invariant to a common per-row scale, so either matrix gives the same answer --
    the raw one is used because it is what the definition is stated on.
    """
    A = A.tocsr().astype(np.float64)
    n = A.shape[0]
    self_w = A.diagonal()
    M = np.zeros((n, n_clusters))
    sw = np.zeros(n)
    indptr, indices, data = A.indptr, A.indices, A.data
    for i in range(n):
        sl = slice(indptr[i], indptr[i + 1])
        cols, vals = indices[sl], data[sl]
        m = cols != i
        if m.any():
            np.add.at(M[i], code[cols[m]], vals[m])
        sw[i] = self_w[i]
    tot = np.maximum(M.sum(axis=1) + sw, EPS)
    return M / tot[:, None], sw / tot


def _frac_chunk(M, sw, own, P, j0, j1):
    """frac[i, g] over a gene slice -- the one definition, never materialized whole."""
    F = M @ P[:, j0:j1]
    O = sw[:, None] * P[own, j0:j1]
    return F / np.maximum(F + O, EPS)


def compute_drift(
    X,
    A,
    design: npt.NDArray[np.floating],
    var_names,
    clusters: npt.NDArray | None = None,
    weights: str = "none",
    resolution: float = 1.0,
    n_pcs: int = 50,
    n_top_genes: int = 2000,
    gene_chunk: int = 2000,
    seed: int = 0,
    verbose: bool = True,
):
    """The six steps above. Returns (u [ncov, ngenes], cluster labels [ncells]).

    `X` is the segmented count matrix, `A` the raw transition matrix, `design` the
    dense design matrix in the SAME row order -- which is the whole reason this can
    live inside `Regression`: there is no join to get wrong.
    """
    if weights not in ("none", "size", "irls"):
        raise ValueError(
            f"drift_weights must be one of 'none', 'size', 'irls'; got {weights!r}")
    X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)
    ncells, ngenes = X.shape
    if A.shape[0] != ncells:
        raise ValueError(
            f"the transition matrix is {A.shape[0]} x {A.shape[1]} but there are "
            f"{ncells} cells")
    if design.shape[0] != ncells:
        raise ValueError(
            f"the design has {design.shape[0]} rows but there are {ncells} cells")

    if clusters is None:
        clusters = leiden_clusters(X, var_names, resolution, n_pcs, n_top_genes, seed)
    clusters = np.asarray(clusters)
    if clusters.shape != (ncells,):
        raise ValueError(
            f"`drift_clusters` has shape {clusters.shape}, expected ({ncells},) -- "
            "one label per cell, in this object's row order")
    levels = pd.Index(sorted(set(clusters.tolist())))
    code = levels.get_indexer(clusters)
    if verbose:
        print(f"drift: {len(levels)} clusters over {ncells} cells", flush=True)

    P = cluster_profiles(X, code, len(levels))
    M, sw = donor_mixture(A, code, len(levels))
    own = code

    Xd = np.asarray(design, dtype=np.float64)
    counts_i = np.asarray(X.sum(axis=1)).ravel().astype(float)
    w_size = np.maximum(counts_i, 1.0)
    w = {"none": None, "size": w_size, "irls": w_size}[weights]

    rg = pg = None
    if weights == "irls":
        rg = moment_dispersion(X)
        pg = np.asarray(X.mean(axis=0)).ravel() / max(counts_i.mean(), EPS)

    u = np.empty((Xd.shape[1], ngenes))
    n_sat = 0
    for j0 in range(0, ngenes, gene_chunk):
        j1 = min(j0 + gene_chunk, ngenes)
        frac = _frac_chunk(M, sw, own, P, j0, j1)
        u[:, j0:j1] = project(Xd, frac, w)
        if weights == "irls":
            for j in range(j0, j1):
                mu = counts_i * pg[j]
                if mu.max() < 0.1 * rg[j]:
                    continue
                wj = mu * rg[j] / (rg[j] + mu)
                u[:, j] = project(Xd, frac[:, j - j0:j - j0 + 1], wj).ravel()
                n_sat += 1
    if weights == "irls" and verbose:
        print(f"drift: {n_sat} of {ngenes} genes recomputed with NB saturation "
              f"({100 * n_sat / max(ngenes, 1):.1f}%)", flush=True)

    if not np.all(np.isfinite(u)):
        warnings.warn(
            f"{int((~np.isfinite(u)).sum())} non-finite entries in the computed "
            "drift covariate; they are set to 0. A design column with no support, "
            "or a cluster with no cells, is the usual cause.", stacklevel=2)
        u = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
    return u, clusters
