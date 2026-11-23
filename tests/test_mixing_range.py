"""The segmenter's mixing as a nuisance with a declared range.

A_s = (1 - s) I + s A interpolates between no mixing (s = 0) and the segmenter's
A (s = 1). With `mixing_range`, the interval is the union of refits over the
range: the bar is that rows stay stochastic, a zero-width range reproduces the
ordinary interval, and a [0, 1] range covers full fits at either end.
"""

import numpy as np
import pytest
import torch
from anndata import AnnData
from scipy.sparse import csr_matrix

from segreg.regression import Regression

NCELLS, NGENES = 800, 40
FORMULA = "~ 1 + exposure + C(clust)"


def make_data(seed=0):
    """Two clusters with different profiles under a known mixing (70% self, 30%
    spread over 4 random neighbours), so the mixing is informative."""
    rng = np.random.default_rng(seed)
    genes = [f"g{j}" for j in range(NGENES)]
    exposure = rng.normal(0, 1, NCELLS)
    log_size = np.log(rng.integers(800, 1200, NCELLS).astype(float))
    cl = (rng.random(NCELLS) < 0.5).astype(np.int64)
    prof = rng.random((2, NGENES)) + 0.2
    prof /= prof.sum(axis=1, keepdims=True)
    beta = np.zeros(NGENES)
    beta[rng.choice(NGENES, 6, replace=False)] = rng.choice([-0.9, -0.5, 0.5, 0.9], 6)
    lam = np.exp(exposure[:, None] * beta[None, :] + log_size[:, None]) * prof[cl] * 0.5

    nb = np.stack([rng.choice(NCELLS, 4, replace=False) for _ in range(NCELLS)])
    rows = np.repeat(np.arange(NCELLS), 5)
    cols = np.concatenate([np.arange(NCELLS)[:, None], nb], axis=1).ravel()
    vals = np.tile([0.70] + [0.075] * 4, NCELLS)
    A = csr_matrix((vals, (rows, cols)), shape=(NCELLS, NCELLS))
    X = rng.poisson(np.asarray(A @ lam)).astype(np.float32)

    tot = np.asarray(A.sum(axis=1)).ravel()
    counts = X.sum(axis=1)
    ad = AnnData(X=csr_matrix(X))
    ad.var_names = genes
    ad.obs["exposure"] = exposure
    ad.obs["clust"] = cl.astype(str)
    # segreg normalizes by (rowsum + to_bg), so scale to a counts-space matrix
    ad.obsp["state_transitions"] = csr_matrix(A.multiply(counts[:, None]))
    ad.obs["to_bg_trans_count"] = counts * (1.0 - tot)
    return ad, beta


def _fit(ad, device=None, nepochs=600, **kw):
    reg = Regression(ad, FORMULA, include_mixing=True, **kw)
    reg.fit(nepochs=nepochs, batch_size=400, seed=0, verbose=False, compile=False,
            β_prior="normal", device=device)
    c = reg.get_regression_coefficients(credible_interval=0.95).query("Covariate == 'exposure'")
    return reg, c.set_index("Gene").reindex(ad.var_names)


def test_scaled_mixing_rows_stay_STOCHASTIC():
    ad, _ = make_data()
    reg = Regression(ad, FORMULA, include_mixing=True)
    full = np.asarray(reg._A_full.sum(1)).ravel() + reg._bg_full
    for s in (0.0, 0.3, 1.0):
        A, bg = reg._mixing_at(s)
        assert np.allclose(np.asarray(A.sum(1)).ravel() + bg, full, atol=1e-5)
    A0, bg0 = reg._mixing_at(0.0)
    assert np.allclose(A0.diagonal(), 1.0) and A0.nnz == A0.shape[0] and np.all(bg0 == 0)


def test_a_ZERO_width_mixing_range_is_the_ordinary_interval():
    ad, _ = make_data()
    _, lin = _fit(ad)
    _, ref = _fit(ad, mixing_scale=1.0, mixing_range=(1.0, 1.0))
    w_lin, w_ref = lin["Upper"] - lin["Lower"], ref["Upper"] - ref["Lower"]
    assert np.allclose(w_ref, w_lin, rtol=0.05)


def test_a_full_mixing_range_COVERS_both_ends():
    ad, _ = make_data()
    _, ref = _fit(ad, mixing_scale=0.5, mixing_range=(0.0, 1.0))
    for s in (0.0, 1.0):
        _, c = _fit(ad, mixing_scale=s)
        b = c["Mean"].to_numpy()
        assert ((ref["Lower"].to_numpy() <= b + 0.02) & (b - 0.02 <= ref["Upper"].to_numpy())).all()


@pytest.mark.parametrize("kw, match", [
    ({"mixing_scale": 1.5}, "mixing_scale must be in"),
    ({"mixing_scale": 0.5, "mixing_range": (0.6, 1.0)}, "mixing_range must satisfy"),
    ({"mixing_scale": 0.5, "mixing_range": (0.0, 1.2)}, "mixing_range must satisfy"),
    ({"include_mixing": False, "mixing_range": (0.0, 1.0)}, "needs include_mixing"),
])
def test_bad_mixing_arguments_are_ERRORS(kw, match):
    ad, _ = make_data()
    kw = {"include_mixing": True, **kw}
    with pytest.raises(ValueError, match=match):
        Regression(ad, FORMULA, **kw)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("kw", [{}, {"mixing_scale": 0.5, "mixing_range": (0.0, 1.0)}])
def test_intervals_run_on_CUDA(kw):
    """posterior_sd and range_interval have device-specific paths (chunk sizing
    from free GPU memory) that the CPU tests never reach."""
    ad, _ = make_data()
    _, c = _fit(ad, device="cuda", nepochs=5, **kw)
    assert np.isfinite(c[["Lower", "Upper"]].to_numpy()).all()
    assert (c["Lower"] <= c["Upper"]).all()
