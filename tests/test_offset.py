"""The externally-supplied offset seam.

segreg's offset is the observed total, which is what the segmenter ASSIGNED to a cell
(retained + inflow) rather than what the cell EMITTED (retained + outflow). The seam
lets a caller pass the latter. These tests check the SEAM, not the idea -- whether a
flow-corrected offset helps is what bench/ measures (simple-seg-sim DESIGN_v2 §95).

The bar for a seam: default unchanged, a correct offset recovers the truth better than
a wrong one, and every way of misaligning it is an error rather than a silent fill.
"""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy.sparse import csr_matrix, diags

from segreg.regression import Regression

NCELLS, NGENES = 800, 40


def make(seed=0, depth=1.5, scale_err=0.0):
    """Poisson counts at a known per-cell size factor.

    `scale_err` multiplies each cell's TRUE size factor by a random factor before the
    counts are drawn, so `log(observed total)` is a biased offset and the true one is
    recoverable -- which is the situation the seam exists for.
    """
    rng = np.random.default_rng(seed)
    exposure = rng.normal(0.0, 1.0, NCELLS)
    true_size = rng.integers(1500, 2500, NCELLS).astype(float)
    beta = rng.choice([-0.6, 0.0, 0.6], NGENES)
    gene_bias = np.full(NGENES, np.log(depth) - np.log(2000.0))
    lam = np.exp(exposure[:, None] * beta[None, :] + gene_bias[None, :]
                 + np.log(true_size)[:, None])
    if scale_err:
        # A per-cell multiplicative distortion of the REALISED counts, so the observed
        # total is wrong by it while `true_size` remains the right offset.
        lam = lam * np.exp(rng.normal(0.0, scale_err, NCELLS))[:, None]
    X = rng.poisson(lam).astype(np.float32)
    adata = AnnData(X=csr_matrix(X))
    adata.var_names = [f"g{j}" for j in range(NGENES)]
    adata.obs["exposure"] = exposure
    adata.obs["original_cell_id"] = np.arange(NCELLS, dtype=np.int64)
    adata.obsp["state_transitions"] = csr_matrix(
        diags(np.asarray(X.sum(axis=1)).ravel()))
    adata.obs["to_bg_trans_count"] = np.zeros(NCELLS)
    return adata, beta, true_size


def test_default_is_UNCHANGED_and_is_the_observed_total():
    adata, _, _ = make()
    r = Regression(adata, "~ 1 + exposure", include_mixing=True)
    want = np.log(np.maximum(np.asarray(adata.X.sum(axis=1)).squeeze(), 1))
    assert np.allclose(r.log_size, want.astype(np.float32))
    assert r.offset_supplied is False


def test_a_supplied_offset_REPLACES_log_size_by_vector_and_by_Series():
    adata, _, true_size = make()
    want = np.log(true_size)
    a = Regression(adata, "~ 1 + exposure", offset=want)
    assert np.allclose(a.log_size, want.astype(np.float32))
    assert a.offset_supplied is True
    # A Series is aligned by original_cell_id, not by position -- shuffle it to prove
    # the alignment is real rather than incidental.
    idx = np.asarray(adata.obs["original_cell_id"])
    perm = np.random.default_rng(1).permutation(len(idx))
    s = pd.Series(want[perm], index=idx[perm])
    b = Regression(adata, "~ 1 + exposure", offset=s)
    assert np.allclose(b.log_size, a.log_size)


def test_the_RIGHT_offset_beats_the_observed_total_when_the_total_is_BIASED():
    """The seam earns its place only if a better offset gives a better coefficient.

    With a per-cell distortion of the counts, log(observed total) absorbs part of the
    exposure signal; the true size factor does not.
    """
    adata, beta, true_size = make(scale_err=0.4)
    def err(**kw):
        r = Regression(adata, "~ 1 + exposure", include_mixing=True, **kw)
        r.fit(nepochs=600, batch_size=400, seed=0, verbose=False, compile=False,
              β_prior="normal")
        d = r.get_regression_coefficients()
        est = (d[d["Covariate"] == "exposure"].set_index("Gene")["Mean"]
               .reindex(adata.var_names).to_numpy())
        return float(np.abs(est - beta).mean())
    e_obs = err()
    e_true = err(offset=np.log(true_size))
    assert e_true < e_obs, f"true offset {e_true:.4f} vs observed total {e_obs:.4f}"


def test_every_MISALIGNMENT_is_an_error_not_a_silent_fill():
    adata, _, true_size = make()
    with pytest.raises(ValueError, match="expected"):
        Regression(adata, "~ 1 + exposure", offset=np.log(true_size)[:-5])
    with pytest.raises(ValueError, match="non-finite"):
        bad = np.log(true_size).copy(); bad[3] = np.nan
        Regression(adata, "~ 1 + exposure", offset=bad)
    with pytest.raises(ValueError, match="missing"):
        s = pd.Series(np.log(true_size), index=np.arange(NCELLS) + 10_000)
        Regression(adata, "~ 1 + exposure", offset=s)
