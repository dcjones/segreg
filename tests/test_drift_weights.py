"""The `mu` drift weight tier: a gene-specific weighted projection."""

import numpy as np
import scipy.sparse as sp

from segreg import drift as D


def test_project_per_gene_matches_a_direct_weighted_solve():
    rng = np.random.default_rng(0)
    n, k, g = 400, 5, 7
    Xd = rng.normal(size=(n, k))
    Y = rng.normal(size=(n, g))
    W = rng.gamma(2.0, size=(n, g))
    got = D.project_per_gene(Xd, Y, W, D.design_pairs(Xd), ridge=0.5)
    for j in range(g):
        gram = Xd.T @ (Xd * W[:, j:j + 1]) + 0.5 * np.eye(k)
        want = np.linalg.solve(gram, Xd.T @ (W[:, j] * Y[:, j]))
        np.testing.assert_allclose(got[:, j], want, rtol=1e-10, atol=1e-12)


def make_inputs(seed=0, n=300, ngenes=12, nclusters=3):
    rng = np.random.default_rng(seed)
    clusters = rng.integers(0, nclusters, size=n).astype(str)
    X = sp.csr_matrix(rng.poisson(rng.gamma(1.0, 2.0, size=(n, ngenes))))
    # a sparse, row-dominant transition matrix: mostly self, some neighbours
    rows, cols, vals = [], [], []
    for i in range(n):
        rows.append(i); cols.append(i); vals.append(20)
        for j in rng.choice(n, size=3, replace=False):
            if j != i:
                rows.append(i); cols.append(j); vals.append(int(rng.integers(1, 5)))
    A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    design = np.column_stack([np.ones(n), rng.normal(size=n), rng.normal(size=n)])
    return X, A, design, clusters


def test_mu_tier_is_the_expected_count_weighted_projection():
    X, A, design, clusters = make_inputs()
    ngenes = X.shape[1]
    u, _ = D.compute_drift(X, A, design, [f"g{j}" for j in range(ngenes)],
                           clusters=clusters, weights="mu", transform="log",
                           mu_ridge=2.0, gene_chunk=5, verbose=False)
    # brute force, straight from the definitions
    code = np.unique(clusters, return_inverse=True)[1]
    P = D.cluster_profiles(X, code, code.max() + 1)
    M, sw = D.donor_mixture(A, code, code.max() + 1)
    counts = np.asarray(X.sum(axis=1)).ravel()
    for j in range(ngenes):
        F = M @ P[:, j]
        S = np.maximum(F + sw * P[code, j], D.EPS)
        e = -np.log1p(-np.minimum(F / S, 1 - 1e-6))
        w = counts * S
        gram = design.T @ (design * w[:, None]) + 2.0 * np.eye(design.shape[1])
        want = np.linalg.solve(gram, design.T @ (w * e))
        np.testing.assert_allclose(u[:, j], want, rtol=1e-8, atol=1e-10)


def test_the_existing_tiers_are_unchanged_by_the_refactor():
    X, A, design, clusters = make_inputs(seed=1)
    names = [f"g{j}" for j in range(X.shape[1])]
    u, _ = D.compute_drift(X, A, design, names, clusters=clusters, verbose=False)
    code = np.unique(clusters, return_inverse=True)[1]
    P = D.cluster_profiles(X, code, code.max() + 1)
    M, sw = D.donor_mixture(A, code, code.max() + 1)
    frac = D._frac_chunk(M, sw, code, P, 0, X.shape[1])
    np.testing.assert_array_equal(u, D.project(design, frac))
