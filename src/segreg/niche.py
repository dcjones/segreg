"""Spatial-niche design features for niche differential expression.

A cell's *niche* is summarized by averaging a reduced-dimensionality expression
representation over its spatial neighbors. By default the representation is
proseg's NMF metagene rates (``obsm["metagene_rates"]``), but any ``obsm`` key or
an explicit array can be supplied.

Feeding these neighbor-averaged features to :class:`~segreg.RegressionModel` as
covariates fits, per gene, how a cell's own expression responds to its
environment. Because the model keeps contamination in its additive inflow term
(``alpha_g * inflow_cg``) and the niche response in the own-signal rate
(``D @ B``), the recovered niche coefficients are the part of the
neighbor-correlated signal that segmentation bleed does *not* explain -- no
changes to the model are required, only this contrived design matrix.

The typical entry point is :func:`add_niche_covariates`, which computes the
features, writes them onto ``adata.obs``, and returns a patsy formula ready to
hand to ``RegressionModel``.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree


def _neighbor_adjacency(
    coords: np.ndarray,
    n_neighbors: int | None,
    radius: float | None,
    weighting: str,
    length_scale: float | None,
    include_self: bool,
) -> csr_matrix:
    """Row-normalized sparse neighbor-averaging operator A (N x N).

    ``A @ feats`` yields, for each cell, the (optionally distance-weighted) mean
    of its neighbors' features. Rows with no neighbors are all-zero, so those
    cells get a zero niche vector.
    """
    n = coords.shape[0]
    tree = cKDTree(coords)

    if radius is not None:
        neighbors = tree.query_ball_point(coords, r=radius, workers=-1)
        counts = np.fromiter((len(nb) for nb in neighbors), dtype=np.int64, count=n)
        cols = np.fromiter(
            itertools.chain.from_iterable(neighbors),
            dtype=np.int64,
            count=int(counts.sum()),
        )
        rows = np.repeat(np.arange(n), counts)
        d = np.linalg.norm(coords[rows] - coords[cols], axis=1)
    elif n_neighbors is not None:
        # query one extra so we can drop self (the nearest point at distance 0).
        dist, idx = tree.query(coords, k=n_neighbors + 1, workers=-1)
        rows = np.repeat(np.arange(n), n_neighbors + 1)
        cols = idx.reshape(-1)
        d = dist.reshape(-1)
    else:
        raise ValueError("Specify exactly one of n_neighbors or radius.")

    if not include_self:
        keep = cols != rows
        rows, cols, d = rows[keep], cols[keep], d[keep]

    if weighting == "gaussian":
        if length_scale is None:
            positive = d[d > 0]
            length_scale = float(np.median(positive)) if positive.size else 1.0
        w = np.exp(-(d**2) / (2.0 * length_scale**2))
    elif weighting == "uniform":
        w = np.ones_like(d)
    else:
        raise ValueError("weighting must be 'uniform' or 'gaussian'.")

    A = csr_matrix((w, (rows, cols)), shape=(n, n))
    row_sums = np.asarray(A.sum(axis=1)).squeeze()
    inv = np.zeros_like(row_sums)
    nz = row_sums > 0
    inv[nz] = 1.0 / row_sums[nz]
    # scale each row by its inverse degree
    A = csr_matrix((A.data * np.repeat(inv, np.diff(A.indptr)), A.indices, A.indptr),
                   shape=A.shape)
    return A


def build_niche_features(
    adata: AnnData,
    feature_key: str = "metagene_rates",
    features: np.ndarray | None = None,
    spatial_key: str = "spatial",
    n_neighbors: int | None = 15,
    radius: float | None = None,
    weighting: str = "uniform",
    length_scale: float | None = None,
    include_self: bool = False,
    standardize: bool = True,
    prefix: str = "niche",
) -> pd.DataFrame:
    """Average a per-cell feature representation over spatial neighbors.

    Parameters
    ----------
    adata
        AnnData with cell coordinates in ``obsm[spatial_key]`` and, unless
        ``features`` is given, the representation in ``obsm[feature_key]``.
    feature_key
        ``obsm`` key of the representation to aggregate (default proseg's
        ``"metagene_rates"``).
    features
        Explicit (N, K) representation, overriding ``feature_key``. Use this to
        supply a different feature set (e.g. corrected-expression PCs).
    n_neighbors, radius
        Neighborhood definition. Give exactly one: ``n_neighbors`` for a fixed
        k-NN graph, ``radius`` for a fixed spatial scale (variable degree).
    weighting, length_scale
        ``"uniform"`` mean, or ``"gaussian"`` distance weighting with the given
        length scale (default: median neighbor distance).
    include_self
        Whether the focal cell contributes to its own niche (default False, so a
        niche coefficient cannot regress a cell's expression on itself).
    standardize
        Z-score each niche column so coefficients are per-SD and the regression
        is well conditioned. With standardization the columns are centered, so
        pair this with an intercept (``RegressionModel``'s ``gene_bias``
        provides one) rather than a patsy ``Intercept``.

    Returns
    -------
    DataFrame indexed by ``adata.obs_names`` with one column per feature.
    """
    if (n_neighbors is None) == (radius is None):
        raise ValueError("Specify exactly one of n_neighbors or radius.")

    coords = np.asarray(adata.obsm[spatial_key], dtype=np.float64)

    if features is None:
        feats = np.asarray(adata.obsm[feature_key], dtype=np.float64)
    else:
        feats = np.asarray(features, dtype=np.float64)
    if feats.shape[0] != coords.shape[0]:
        raise ValueError(
            f"features has {feats.shape[0]} rows but there are {coords.shape[0]} cells."
        )

    A = _neighbor_adjacency(
        coords, n_neighbors, radius, weighting, length_scale, include_self
    )
    niche = np.asarray(A @ feats)

    if standardize:
        mean = niche.mean(axis=0, keepdims=True)
        std = niche.std(axis=0, keepdims=True)
        std[std == 0] = 1.0
        niche = (niche - mean) / std

    names = [f"{prefix}_{i:02d}" for i in range(niche.shape[1])]
    return pd.DataFrame(niche, index=adata.obs_names, columns=names)


def add_niche_covariates(
    adata: AnnData,
    include_intercept: bool = False,
    **kwargs,
) -> str:
    """Compute niche features, attach them to ``adata.obs``, and return a formula.

    ``**kwargs`` are forwarded to :func:`build_niche_features`. The returned
    string is a patsy formula listing the niche columns, suitable for
    ``RegressionModel(adata, formula)``. By default no patsy intercept is added
    (the model's per-gene ``gene_bias`` already serves as one); set
    ``include_intercept=True`` to add one anyway.
    """
    niche_df = build_niche_features(adata, **kwargs)
    adata.obs[niche_df.columns] = niche_df
    lead = "1" if include_intercept else "0"
    return " + ".join([lead, *niche_df.columns])
