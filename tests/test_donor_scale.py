"""A learned scale on A's off-diagonal, indexed by the donor's cluster.

Data are generated WITH a known per-cluster distortion of the mixing matrix: proseg's
"estimate" is the truth with cluster 1's donations inflated, so the correct u is
known and recovery is testable. That is the right first bar -- if the scale cannot be
recovered when the distortion is exactly of the modelled form, no clustering scheme
will save it.
"""

import numpy as np
import pytest
from anndata import AnnData
from scipy.sparse import csr_matrix

from segreg.regression import Regression

NCELLS, NGENES = 800, 40
U_TRUE = np.log(1.8)          # cluster 1 donates 1.8x what proseg thinks


def make_data(seed=0, distort=True):
    """Two donor clusters with different profiles, so the mixing is informative."""
    rng = np.random.default_rng(seed)
    genes = [f"g{j}" for j in range(NGENES)]
    exposure = rng.normal(0, 1, NCELLS)
    log_size = np.log(rng.integers(800, 1200, NCELLS).astype(float))
    cl = (rng.random(NCELLS) < 0.5).astype(np.int64)      # donor cluster per cell
    # cluster-specific expression, so which donor a cell draws from MATTERS
    prof = rng.random((2, NGENES)) + 0.2
    prof /= prof.sum(axis=1, keepdims=True)
    beta = np.zeros(NGENES)
    # MIXED signs and magnitudes: with a constant effect size the recovery
    # correlation below is 0/0.
    beta[rng.choice(NGENES, 6, replace=False)] = rng.choice([-0.9, -0.5, 0.5, 0.9], 6)
    lam = np.exp(exposure[:, None] * beta[None, :] + log_size[:, None]) \
        * prof[cl] * 0.5
    # A TRUE mixing: 70% self, 30% spread over 4 random neighbours
    nb = np.stack([rng.choice(NCELLS, 4, replace=False) for _ in range(NCELLS)])
    rows, cols, vals = [], [], []
    for i in range(NCELLS):
        rows += [i] * 5; cols += [i] + list(nb[i]); vals += [0.70] + [0.075] * 4
    A_true = csr_matrix((vals, (rows, cols)), shape=(NCELLS, NCELLS))
    lam_obs = np.asarray(A_true @ lam)
    X = rng.poisson(lam_obs).astype(np.float32)

    # what the segmenter "reports": cluster 1's donations DEFLATED by 1/1.8, so the
    # model must learn u = +log(1.8) to get back to the truth.
    d = np.array(vals, dtype=float).copy()
    if distort:
        for k, (i, j) in enumerate(zip(rows, cols)):
            if i != j and cl[j] == 1:
                d[k] /= 1.8
    A_est = csr_matrix((d, (rows, cols)), shape=(NCELLS, NCELLS))
    tot = np.asarray(A_est.sum(axis=1)).ravel()
    counts = np.asarray(X.sum(axis=1)).ravel()
    ad = AnnData(X=csr_matrix(X)); ad.var_names = genes
    ad.obs["exposure"] = exposure
    # The donor clusters have DIFFERENT expression profiles -- that is what makes the
    # mixing informative about u -- so the cluster has to be in the design too.
    # Without it the model cannot represent those levels at all and u absorbs the
    # gap by shifting the donor mixture, which is a confound in the GENERATOR rather
    # than a property of the scale. (Measured: u drifted to >0.25 with no distortion
    # present until the cluster term was added.)
    ad.obs["clust"] = cl.astype(str)
    # segreg normalizes by (rowsum + to_bg), so scale the counts-space matrix here
    ad.obsp["state_transitions"] = csr_matrix(A_est.multiply(counts[:, None]))
    ad.obs["to_bg_trans_count"] = counts * (1.0 - tot)
    return ad, cl, beta


def fit(ad, cl, u_prior=0.5, nepochs=800, seed=0):
    reg = Regression(ad, "~ 1 + exposure + C(clust)", include_mixing=True,
                     include_donor_scale=cl is not None, donor_clusters=cl)
    reg.fit(nepochs=nepochs, batch_size=400, seed=seed, verbose=False, compile=False,
            β_prior="normal", u_prior_σ=u_prior)
    return reg


def test_the_scale_RECOVERS_a_known_per_cluster_distortion():
    ad, cl, _ = make_data()
    reg = fit(ad, cl)
    d = reg.get_donor_scales().set_index("cluster")
    # cluster 0 undistorted, cluster 1 deflated by 1.8 -> u1 - u0 should be log(1.8)
    got = d.loc[1, "u"] - d.loc[0, "u"]
    assert got == pytest.approx(U_TRUE, abs=0.35), (
        f"recovered u1-u0 = {got:.3f} against a true {U_TRUE:.3f}\n{d}")


def test_MASS_is_conserved_so_the_background_share_cannot_move():
    """§55: slack is worth 15x the matrix, so a matrix experiment that perturbs the
    background share is measuring the background share. Rows must keep their total."""
    ad, cl, _ = make_data()
    reg = fit(ad, cl, nepochs=200)
    before = np.asarray(reg.A.sum(axis=1)).ravel()
    after = np.asarray(reg._scaled_A().sum(axis=1)).ravel()
    assert np.abs(after - before).max() < 1e-5, "row totals moved"


def test_it_is_a_NO_OP_when_there_is_nothing_to_correct():
    """Undistorted data: u should stay near 0 rather than drifting off to explain
    noise, which is the failure mode a weakly-identified nuisance shows (§69)."""
    ad, cl, _ = make_data(distort=False)
    reg = fit(ad, cl)
    d = reg.get_donor_scales()
    assert np.abs(d["u"].to_numpy()).max() < 0.25, f"u drifted with nothing to fix:\n{d}"


def test_it_does_not_eat_the_exposure_signal():
    ad, cl, beta = make_data()
    reg = fit(ad, cl)
    df = reg.get_regression_coefficients()
    est = (df[df["Covariate"] == "exposure"].set_index("Gene")["Mean"]
           .reindex(ad.var_names).to_numpy())
    idx = beta != 0
    assert np.corrcoef(est[idx], beta[idx])[0, 1] > 0.8
    assert np.abs(est[~idx]).mean() < 0.3


def test_missing_clusters_is_an_error_not_a_silent_noop():
    ad, cl, _ = make_data()
    with pytest.raises(ValueError, match="requires `donor_clusters`"):
        Regression(ad, "~ 1 + exposure + C(clust)", include_donor_scale=True)
    with pytest.raises(ValueError, match="include_donor_scale is False"):
        Regression(ad, "~ 1 + exposure + C(clust)", donor_clusters=cl)
