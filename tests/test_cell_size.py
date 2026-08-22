"""A per-cell log-scale learned as a deviation from log(observed total).

The generator makes the OFFSET WRONG in the one way that matters: `mu = A lambda`
means a cell's predicted total is a weighted average of its NEIGHBOURS' totals, so
wherever totals do not vary smoothly in space, log(observed total) is not the scale
the mixing model wants. Here neighbourhoods are deliberately built to mix cells of
very different depth, the true rates are known, and the test asks whether the fitted
delta moves toward the scale that makes the model total-consistent.

simple-seg-sim DESIGN_v2 §74 is why this exists: on the 16.7k-gene panel the spread of
that mismatch made mixing fit WORSE than no mixing by 37.7 nats/cell.
"""

import numpy as np
import pytest
from anndata import AnnData
from scipy.sparse import csr_matrix

from segreg.regression import Regression

NCELLS, NGENES = 600, 30


def make_data(seed=0, spread=1.4):
    """Neighbourhoods that mix depths, so log(observed total) is the WRONG offset."""
    rng = np.random.default_rng(seed)
    genes = [f"g{j}" for j in range(NGENES)]
    exposure = rng.normal(0, 1, NCELLS)
    # depth varies a LOT and is arranged so neighbours differ: alternating tiers.
    tier = np.arange(NCELLS) % 2
    depth = np.exp(rng.normal(0, 0.2, NCELLS) + spread * tier)
    size = 400.0 * depth
    prof = rng.random(NGENES) + 0.2
    prof /= prof.sum()
    beta = np.zeros(NGENES)
    beta[rng.choice(NGENES, 5, replace=False)] = rng.choice([-0.8, 0.8], 5)
    lam = np.exp(exposure[:, None] * beta[None, :]) * prof[None, :] * size[:, None]
    # each cell donates to 4 neighbours drawn across tiers, so A mixes depths
    nb = np.stack([rng.choice(NCELLS, 4, replace=False) for _ in range(NCELLS)])
    rows, cols, vals = [], [], []
    for i in range(NCELLS):
        rows += [i] * 5; cols += [i] + list(nb[i]); vals += [0.68] + [0.08] * 4
    A = csr_matrix((vals, (rows, cols)), shape=(NCELLS, NCELLS))
    X = rng.poisson(np.asarray(A @ lam)).astype(np.float32)
    counts = np.asarray(X.sum(axis=1)).ravel()
    ad = AnnData(X=csr_matrix(X)); ad.var_names = genes
    ad.obs["exposure"] = exposure
    ad.obsp["state_transitions"] = csr_matrix(A.multiply(counts[:, None]))
    ad.obs["to_bg_trans_count"] = counts * (1.0 - np.asarray(A.sum(axis=1)).ravel())
    return ad, size, counts


def fit(ad, cell_size, σ=0.5, nepochs=600, seed=0):
    reg = Regression(ad, "~ 1 + exposure", include_mixing=True,
                     include_cell_size=cell_size)
    reg.fit(nepochs=nepochs, batch_size=300, seed=seed, verbose=False, compile=False,
            β_prior="normal", size_prior_σ=σ)
    return reg


def test_delta_moves_toward_the_TOTAL_CONSISTENT_scale():
    """The fitted offset should track the cell's TRUE size better than the observed
    total does -- that is the whole claim."""
    ad, size, counts = make_data()
    reg = fit(ad, cell_size=True)
    d = reg.get_cell_sizes()
    truth = np.log(size)
    truth = truth - truth.mean()
    fixed = d["log_size_fixed"].to_numpy(); fixed = fixed - fixed.mean()
    fitted = d["log_size_fitted"].to_numpy(); fitted = fitted - fitted.mean()
    err_fixed = np.abs(fixed - truth).mean()
    err_fitted = np.abs(fitted - truth).mean()
    assert err_fitted < err_fixed, (
        f"fitted offset error {err_fitted:.4f} did not beat the fixed offset's "
        f"{err_fixed:.4f}\n{d.head()}")


def test_it_is_OFF_by_default_and_delta_zero_reproduces_the_fixed_offset():
    ad, _, _ = make_data()
    reg = fit(ad, cell_size=False)
    assert not reg.include_cell_size
    with pytest.raises(Exception):
        reg.get_cell_sizes()


def test_a_TIGHT_prior_collapses_onto_the_fixed_offset():
    """size_prior_σ is the whole knob: at ~0 the arm must be the status quo."""
    ad, _, _ = make_data()
    reg = fit(ad, cell_size=True, σ=1e-3, nepochs=300)
    d = reg.get_cell_sizes()
    assert np.abs(d["delta"]).max() < 0.05, (
        f"delta reached {np.abs(d['delta']).max():.4f} under a 1e-3 prior")
