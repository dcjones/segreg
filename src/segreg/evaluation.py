"""Dataset-agnostic honest-metric harness for decontamination quality.

Motivation: marker-Jaccard at a fixed count cutoff is gameable -- pushing counts
down (or sparsifying via a low-rank reconstruction) lowers cross-type co-occurrence
"for free" without improving biology, and low-rank collinearity inflates
within-type co-expression. This module implements metrics that resist those games:

  * probe_separability -- linear-probe macro-F1 predicting cell type from
    expression. Can only improve if the correction BOTH removes cross-type
    contamination AND preserves within-type signal; zeroing counts hurts it.
    Weakness: relies on (approximate) cell-type labels.

  * leiden_ari -- adjusted Rand index of a standard Leiden clustering vs the
    labels; catches over-fragmentation that a low-rank reconstruction induces.

  * marker_leakage -- scale-invariant, label-robust cross-type contamination:
        leakage(A<-B) = mean[B-markers | cells confidently type A]
                        / mean[B-markers | cells confidently type B]
    averaged over ordered type pairs. A ratio, so uniform count scaling cancels;
    a cross-population comparison, so low-rank collinearity can't inflate it;
    and "zero B everywhere" blows up the denominator, so it can't be gamed by
    sparsifying. Lower = cleaner. Confident cells are taken from the RAW markers.

Cell-type labels are derived from `marker_sets` (a dataset-specific dict mapping
cell type -> list of marker gene names) by argmax of summed marker counts -- a
coarse proxy, which is exactly why separability alone isn't trusted and the
leakage ratio (label-robust) is reported alongside it.

Typical use:
    from segreg.evaluation import evaluate
    adata.layers["corrected"] = model.get_corrected_expression()
    evaluate(adata, {"raw": adata.X, "corrected": adata.layers["corrected"]},
             MARKER_SETS, subset_exclude="tumor")
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, f1_score
from sklearn.model_selection import train_test_split


def marker_labels(
    adata, marker_sets: dict[str, list[str]]
) -> tuple[np.ndarray, list[str], dict[str, list[int]]]:
    """Argmax cell-type label per cell from summed marker counts (on adata.X).

    Returns (labels, celltypes, marker_idx) where labels index into the sorted
    celltypes list and marker_idx maps each type to its marker column indices.
    Marker genes absent from adata.var_names are silently skipped."""
    var = list(adata.var_names)
    celltypes = sorted(marker_sets)
    marker_idx = {
        ct: [var.index(g) for g in marker_sets[ct] if g in var] for ct in celltypes
    }
    X = adata.X
    score = np.zeros((adata.n_obs, len(celltypes)), dtype=np.float64)
    for k, ct in enumerate(celltypes):
        cols = marker_idx[ct]
        if cols:
            score[:, k] = np.asarray(X[:, cols].sum(axis=1)).squeeze()
    return np.argmax(score, axis=1), celltypes, marker_idx


def _as_dense(X) -> np.ndarray:
    return np.asarray(X.todense()) if sparse.issparse(X) else np.asarray(X)


def probe_separability(
    X, labels: np.ndarray, mask: np.ndarray | None = None,
    exclude_cols: np.ndarray | None = None,
    subsample: int = 30000, seed: int = 0, max_iter: int = 300,
) -> float:
    """Macro-F1 of a logistic-regression probe predicting `labels` from log1p(X).

    Scale-robust to overall count level via the model's own fitting, and macro
    averaging weights every cell type equally (so a rare type can't be ignored).
    Subsamples for speed; densifies only the subsample.

    IMPORTANT -- label circularity: `labels` are marker-argmax, so a probe that
    sees the marker genes just echoes them (including their noise), and *any* edit
    to markers -- including correct denoising of noisy low-count reads -- reads as
    "less separable". Pass `exclude_cols` = the marker column indices to drop them
    from the features; the probe then predicts cell type from OTHER genes, which
    measures preserved biology rather than marker fidelity. `evaluate()` does this
    by default. Confirmed on xenium-nsclc: all-genes shows raw≫corrected, but
    non-marker shows corrected ≥ raw."""
    y = labels
    idx = np.arange(len(y))
    if mask is not None:
        idx = idx[mask]
    y = labels[idx]
    if len(idx) > subsample:
        sub, _ = train_test_split(np.arange(len(idx)), train_size=subsample, stratify=y, random_state=seed)
        idx, y = idx[sub], y[sub]
    Xsel = X[idx]
    if exclude_cols is not None:
        keep = np.setdiff1d(np.arange(Xsel.shape[1]), exclude_cols)
        Xsel = Xsel[:, keep]
    Xd = np.log1p(_as_dense(Xsel))
    Xtr, Xte, ytr, yte = train_test_split(Xd, y, test_size=0.3, stratify=y, random_state=seed)
    clf = LogisticRegression(max_iter=max_iter).fit(Xtr, ytr)
    return float(f1_score(yte, clf.predict(Xte), average="macro"))


def leiden_ari(
    X, labels: np.ndarray, mask: np.ndarray | None = None,
    resolution: float = 1.0, n_pcs: int = 50, n_neighbors: int = 15, seed: int = 0,
) -> tuple[int, float, float]:
    """Standard scanpy Leiden workflow; returns (n_clusters, ARI_all, ARI_subset).

    ARI_subset is over `mask` cells only (e.g. non-dominant types), or NaN if no
    mask. Uses flavor='igraph' so leidenalg is not required. scanpy imported
    lazily so it stays an optional dependency."""
    import anndata as ad
    import scanpy as sc

    a = ad.AnnData(X=_as_dense(X).astype(np.float32))
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.pca(a, n_comps=n_pcs)
    sc.pp.neighbors(a, n_neighbors=n_neighbors)
    sc.tl.leiden(a, resolution=resolution, flavor="igraph", n_iterations=2,
                 directed=False, random_state=seed)
    lab = a.obs["leiden"].astype(int).to_numpy()
    ari_all = adjusted_rand_score(labels, lab)
    ari_sub = adjusted_rand_score(labels[mask], lab[mask]) if mask is not None else float("nan")
    return len(np.unique(lab)), float(ari_all), float(ari_sub)


def marker_leakage(
    X, labels: np.ndarray, celltypes: list[str], marker_idx: dict[str, list[int]],
) -> float:
    """Mean over ordered type pairs (A,B) of
        mean[B-markers in type-A cells] / mean[B-markers in type-B cells].
    Lower = less cross-type contamination. Scale-invariant and label-robust; see
    module docstring."""
    conf = {ct: (labels == i) for i, ct in enumerate(celltypes)}
    vals = []
    for B in celltypes:
        cols = marker_idx[B]
        if not cols or conf[B].sum() == 0:
            continue
        sig_B = float(np.asarray(X[conf[B]][:, cols].sum(axis=1)).mean())
        if sig_B <= 0:
            continue
        for A in celltypes:
            if A == B or conf[A].sum() == 0:
                continue
            cont = float(np.asarray(X[conf[A]][:, cols].sum(axis=1)).mean())
            vals.append(cont / sig_B)
    return float(np.mean(vals)) if vals else float("nan")


def evaluate(
    adata, products: dict, marker_sets: dict[str, list[str]],
    subset_exclude: str | None = None, run_leiden: bool = True, seed: int = 0,
) -> pd.DataFrame:
    """Evaluate each named product (name -> cell x gene matrix) on the honest
    metrics and return a DataFrame (also printed).

    Labels come from `marker_sets` applied to adata.X (raw). `subset_exclude`, if
    given (e.g. the dominant/contaminating type like "tumor"), adds probe/Leiden
    columns computed over the remaining cell types only -- the "can we separate
    the harder, non-dominant types" question. `products` should include "raw" to
    get a relative-mass column."""
    labels, celltypes, marker_idx = marker_labels(adata, marker_sets)
    mask = None
    if subset_exclude is not None:
        mask = labels != celltypes.index(subset_exclude)
    # Marker columns are excluded from the probe features to break label
    # circularity (see probe_separability). "probe" = non-marker (primary, honest);
    # "probe_all" = all genes (circular, diagnostic only).
    marker_cols = np.array(sorted({c for cols in marker_idx.values() for c in cols}))

    raw_mass = float(np.asarray(products["raw"].sum())) if "raw" in products else None
    rows = []
    for name, X in products.items():
        rec = {"product": name}
        rec["probe"] = probe_separability(X, labels, exclude_cols=marker_cols, seed=seed)
        rec["probe_all"] = probe_separability(X, labels, seed=seed)
        if mask is not None:
            rec["probe_sub"] = probe_separability(X, labels, mask=mask, exclude_cols=marker_cols, seed=seed)
        if run_leiden:
            nc, ari, ari_sub = leiden_ari(X, labels, mask=mask, seed=seed)
            rec["n_clust"], rec["ari"] = nc, ari
            if mask is not None:
                rec["ari_sub"] = ari_sub
        rec["leakage"] = marker_leakage(X, labels, celltypes, marker_idx)
        if raw_mass:
            rec["mass"] = float(np.asarray(X.sum())) / raw_mass
        rows.append(rec)
    df = pd.DataFrame(rows).set_index("product")
    with pd.option_context("display.float_format", lambda v: f"{v:.4f}"):
        print(df.to_string())
    return df


def decontamination_products(model, batch_size: int = 8192) -> dict[str, sparse.csr_matrix]:
    """Build the product decomposition from a trained FactorizationModel for
    research comparison on the metrics above. Returns csr matrices:
      raw, spatial_inflow = relu(X - a*inflow), spatial_full = that / retention,
      residual = X*lam/mu (== model.get_corrected_expression()), lowrank = lam.
    Isolates where cleaning comes from (spatial inflow/outflow model vs the
    lossy low-rank reconstruction). Requires a diffusion model."""
    import torch

    if not model.model.include_diffusion:
        raise ValueError("decontamination_products requires include_diffusion=True")
    m = model.m
    out = {k: [] for k in ["raw", "spatial_inflow", "spatial_full", "residual", "lowrank"]}
    model.model.eval()
    with torch.no_grad():
        vnorm = model.model.v_norm()
        alpha = torch.exp(model.model.log_α)
        for s in range(0, m, batch_size):
            e = min(s + batch_size, m)
            u = model._encode_chunk(s, e, alpha)
            lam = u @ vnorm
            X = model._chunk_csr_tensor(model.X, s, e).to_dense()
            inflow = model._chunk_csr_tensor(model.inflow, s, e).to_dense()
            phi = model._chunk_csr_tensor(model.φ, s, e).to_dense()
            ret = torch.exp(-alpha.unsqueeze(0) * phi)
            mu = ret * lam + alpha.unsqueeze(0) * inflow
            deltas = alpha.unsqueeze(0) * inflow
            out["raw"].append(sparse.csr_matrix(X.cpu().numpy()))
            out["spatial_inflow"].append(sparse.csr_matrix((X - deltas).clamp(min=0).cpu().numpy()))
            out["spatial_full"].append(sparse.csr_matrix(((X - deltas).clamp(min=0) / ret.clamp(min=1e-3)).cpu().numpy()))
            out["residual"].append(sparse.csr_matrix((X * lam / mu.clamp(min=1e-8)).cpu().numpy()))
            lr = lam.cpu().numpy(); lr[lr < 1e-6] = 0.0
            out["lowrank"].append(sparse.csr_matrix(lr))
    return {k: sparse.vstack(v).tocsr() for k, v in out.items()}
