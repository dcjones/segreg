"""The drift term: does separating beta from delta = kappa*u actually work?

Data are generated FROM the model -- observed coefficients are beta + kappa*u --
so these are self-consistency tests. That is the right first bar: if the split
cannot be recovered when the generative form is exactly right, no amount of
covariate engineering will help. Whether real contamination has this form is what
bench/ measures, not what these tests measure.

The mixing matrix is the identity here (and to_bg_trans_count is 0), so mixing is a
no-op and these tests isolate the drift term.
"""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy.sparse import csr_matrix, diags

from segreg.regression import Regression

NCELLS, NGENES = 900, 60
KAPPA_TRUE = -1.5
N_EFFECT = 6


def make_data(seed=0):
    """Counts whose true exposure coefficient is beta + KAPPA_TRUE * u."""
    rng = np.random.default_rng(seed)
    genes = [f"g{j}" for j in range(NGENES)]

    exposure = rng.normal(0.0, 1.0, NCELLS)
    log_size = np.log(rng.integers(1500, 2500, NCELLS).astype(float))

    # Sparse truth: a handful of real effects, the rest exactly zero. Sparsity is
    # the assumption the split rests on, so the test data has to have it.
    beta = np.zeros(NGENES)
    idx = rng.choice(NGENES, N_EFFECT, replace=False)
    beta[idx] = rng.choice([-0.8, 0.8], N_EFFECT)

    # The drift direction: per-gene, both signs, no relation to beta.
    u = rng.normal(0.0, 0.35, NGENES)

    observed = beta + KAPPA_TRUE * u          # what a naive fit must return
    gene_bias = np.full(NGENES, -np.log(NGENES))
    lam = np.exp(exposure[:, None] * observed[None, :]
                 + gene_bias[None, :] + log_size[:, None])
    X = rng.poisson(lam).astype(np.float32)

    adata = AnnData(X=csr_matrix(X))
    adata.var_names = genes
    adata.obs["exposure"] = exposure
    # Identity mixing: A = I, no background, so mix() is a no-op.
    adata.obsp["state_transitions"] = csr_matrix(
        diags(np.asarray(X.sum(axis=1)).ravel()))
    adata.obs["to_bg_trans_count"] = np.zeros(NCELLS)

    drift = pd.DataFrame(np.zeros((2, NGENES)),
                         index=["Intercept", "exposure"], columns=genes)
    drift.loc["exposure"] = u
    return adata, drift, beta, u, idx


def fit(adata, drift=None, seed=0, nepochs=1200, prior_sigma=1.0):
    reg = Regression(adata, "~ 1 + exposure", include_mixing=True,
                     include_drift=drift is not None, drift=drift)
    reg.fit(nepochs=nepochs, batch_size=300, seed=seed, verbose=False,
            compile=False, β_prior_σ=prior_sigma)
    df = reg.get_regression_coefficients()
    est = (df[df["Covariate"] == "exposure"]
           .set_index("Gene")["Mean"].reindex(adata.var_names).to_numpy())
    return reg, est


def test_without_drift_the_fit_returns_the_BIASED_coefficient():
    """The control. Without the term, the estimate is beta + kappa*u by design."""
    adata, drift, beta, u, _ = make_data()
    _, est = fit(adata)
    target = beta + KAPPA_TRUE * u
    assert np.corrcoef(est, target)[0, 1] > 0.95, "did not recover the biased truth"
    # And the null genes are NOT at zero -- that is the defect being corrected.
    nulls = beta == 0
    assert np.abs(est[nulls]).mean() > 0.2


def test_the_split_is_only_PARTIALLY_identified_and_the_beta_prior_sets_it():
    """MEASURED, and it is the central property of this term.

    beta and delta = kappa*u compete to explain the same variation, and the mean
    model fits the observed coefficients essentially perfectly at ANY split
    (corr(beta_hat + kappa*u, truth) = 0.9998 across every prior tried). So the
    likelihood does not choose; beta's KL prior does, and kappa recovery is a
    monotone function of how tight that prior is:

        beta_prior_sigma  1.0   0.3   0.1   0.05      (kappa_true = -1.5)
        kappa            -0.67 -0.92 -1.35 -1.35
        null |bias|       0.228 0.159 0.108 0.104

    At the DEFAULT sigma of 1.0 the term therefore delivers under half the
    correction it should. beta_prior_sigma is consequently no longer just a
    regularizer -- it sets the correction strength, and it has to be swept as an
    arm parameter rather than left at its default.
    """
    adata, drift, beta, u, _ = make_data()
    nulls = beta == 0

    _, est_off = fit(adata)
    err_off = np.abs(est_off[nulls]).mean()

    kappas, errs = [], []
    for ps in (1.0, 0.3, 0.1):
        reg, est = fit(adata, drift, prior_sigma=ps)
        kappas.append(float(reg.get_drift_scales()["exposure"]))
        errs.append(float(np.abs(est[nulls]).mean()))
        # Whatever the split, the mean model must still fit the observed
        # coefficients -- that is what makes this identification and not misfit.
        recon = est + kappas[-1] * u
        assert np.corrcoef(recon, beta + KAPPA_TRUE * u)[0, 1] > 0.99

    # Tighter prior -> more of the drift attributed to kappa, less residual bias.
    assert kappas[0] > kappas[1] > kappas[2], f"not monotone in prior: {kappas}"
    assert errs[0] > errs[1] > errs[2], f"not monotone in prior: {errs}"
    # At a tight prior the term does most of its job.
    assert kappas[-1] == pytest.approx(KAPPA_TRUE, abs=0.3), f"kappa {kappas[-1]}"
    assert errs[-1] < 0.5 * err_off, f"null |bias| {err_off:.3f} -> {errs[-1]:.3f}"


def test_drift_does_not_eat_the_real_effects():
    """The failure that would make the term useless: explaining beta with kappa."""
    adata, drift, beta, u, idx = make_data()
    _, est = fit(adata, drift, prior_sigma=0.1)
    signs_ok = np.sign(est[idx]) == np.sign(beta[idx])
    assert signs_ok.all(), f"{(~signs_ok).sum()} of {N_EFFECT} effects lost their sign"
    # Attenuation is expected (beta is penalized), but not obliteration.
    assert np.abs(est[idx]).mean() > 0.4 * np.abs(beta[idx]).mean()


def test_misaligned_drift_index_is_an_error_not_a_silent_noop():
    adata, drift, _, _, _ = make_data()
    bad = drift.rename(index={"exposure": "Macrophages"})
    bad = bad.drop(index="Intercept")
    with pytest.raises(ValueError, match="shares no index value"):
        Regression(adata, "~ 1 + exposure", include_drift=True, drift=bad)


def test_drift_without_flag_is_an_error():
    adata, drift, _, _, _ = make_data()
    with pytest.raises(ValueError, match="include_drift is False"):
        Regression(adata, "~ 1 + exposure", drift=drift)


def test_a_prior_on_kappa_takes_the_load_off_the_beta_prior():
    """The point of the kappa prior: recover the correction at the DEFAULT beta prior.

    The likelihood is exactly invariant along (beta + c*u, kappa - c), so something
    has to choose the split. With kappa unpenalized that job falls entirely on
    beta_prior_sigma, which does not transfer across panels. A weakly informative
    N(-1.5, 1) on kappa -- covering the mean -1.46, sd 1.04 measured over 17 groups
    on two datasets -- does the same job with a quantity that does transfer.
    """
    adata, drift, beta, u, _ = make_data()
    nulls = beta == 0

    reg_flat, est_flat = fit(adata, drift)                      # sigma_beta = 1.0
    reg_pri = Regression(adata, "~ 1 + exposure", include_mixing=True,
                         include_drift=True, drift=drift)
    reg_pri.fit(nepochs=1200, batch_size=300, seed=0, verbose=False, compile=False,
                β_prior_σ=1.0, κ_prior_μ=KAPPA_TRUE, κ_prior_σ=1.0)
    df = reg_pri.get_regression_coefficients()
    est_pri = (df[df["Covariate"] == "exposure"]
               .set_index("Gene")["Mean"].reindex(adata.var_names).to_numpy())

    k_flat = float(reg_flat.get_drift_scales()["exposure"])
    k_pri = float(reg_pri.get_drift_scales()["exposure"])
    # At the same (default, loose) beta prior, the kappa prior recovers the
    # correction that beta_prior_sigma alone needed to be tightened 10x to reach.
    assert abs(k_pri - KAPPA_TRUE) < abs(k_flat - KAPPA_TRUE), \
        f"kappa prior did not help: {k_flat} -> {k_pri}"
    assert np.abs(est_pri[nulls]).mean() < np.abs(est_flat[nulls]).mean()


def make_cellspace(adata, drift, rng_seed=1):
    """A field whose design-projected part is exactly `drift`'s direction.

    Two clusters, so frac[i,g] varies by cell and gene through a real donor
    mixture rather than being hand-set -- the point is to exercise the same
    reconstruction the real builder feeds in.
    """
    rng = np.random.default_rng(rng_seed)
    n, G = adata.n_obs, adata.n_vars
    P = rng.random((2, G)) + 0.5
    P = P / P.sum(axis=1, keepdims=True)
    M = np.zeros((n, 2), dtype=np.float32)
    frac_target = 0.15 + 0.10 * (adata.obs["exposure"].to_numpy() > 0)
    M[:, 0] = frac_target
    sw = (1.0 - frac_target).astype(np.float32)
    return dict(M=M, sw=sw, own=np.ones(n, dtype=np.int64), P=P.astype(np.float32),
                original_cell_id=np.asarray(adata.obs["original_cell_id"]),
                genes=np.asarray(list(adata.var_names), dtype=object))


def test_cellspace_term_is_live_and_reconstructs_frac():
    adata, drift, beta, u, _ = make_data()
    adata.obs["original_cell_id"] = np.arange(adata.n_obs, dtype=np.int64)
    cs = make_cellspace(adata, drift)
    reg = Regression(adata, "~ 1 + exposure", include_mixing=True,
                     include_cellspace=True, cellspace=cs)
    reg.fit(nepochs=200, batch_size=300, seed=0, verbose=False, compile=False)
    assert reg.model.include_cellspace
    # kappa moved off its zero init, i.e. the term is doing something.
    assert abs(float(reg.model.κ_cs.detach())) > 1e-3


def test_cellspace_misalignment_is_an_error():
    adata, drift, _, _, _ = make_data()
    adata.obs["original_cell_id"] = np.arange(adata.n_obs, dtype=np.int64)
    cs = make_cellspace(adata, drift)
    cs["original_cell_id"] = cs["original_cell_id"][:-5]
    cs["M"] = cs["M"][:-5]; cs["sw"] = cs["sw"][:-5]; cs["own"] = cs["own"][:-5]
    with pytest.raises(ValueError, match="missing"):
        Regression(adata, "~ 1 + exposure", include_cellspace=True, cellspace=cs)


def test_cellspace_without_flag_is_an_error():
    adata, drift, _, _, _ = make_data()
    adata.obs["original_cell_id"] = np.arange(adata.n_obs, dtype=np.int64)
    with pytest.raises(ValueError, match="include_cellspace is False"):
        Regression(adata, "~ 1 + exposure",
                   cellspace=make_cellspace(adata, drift))


def test_percluster_cellspace_kappa_is_live_and_pooled():
    """Rung 2: one kappa per cluster, with the deviations penalized.

    Checks it is (a) live -- the deviations move off zero -- and (b) actually
    pooled: a tight shrink sigma must pull them back toward the shared level, or
    the cluster count is unchecked freedom against a flat likelihood direction.
    """
    adata, drift, beta, u, _ = make_data()
    adata.obs["original_cell_id"] = np.arange(adata.n_obs, dtype=np.int64)
    cs = make_cellspace(adata, drift)
    spread = {}
    for sig in (1.0, 0.01):
        reg = Regression(adata, "~ 1 + exposure", include_mixing=True,
                         include_cellspace=True, cellspace=cs,
                         cellspace_percluster=True, cellspace_shrink_σ=sig)
        reg.fit(nepochs=200, batch_size=300, seed=0, verbose=False, compile=False)
        k = reg.get_cellspace_scales()
        assert len(k) == cs["P"].shape[0], "expected one kappa per cluster"
        spread[sig] = float(np.std(k))
    assert spread[1.0] > 0, "per-cluster deviations never moved off zero"
    assert spread[0.01] < spread[1.0], (
        f"tight shrinkage did not pool: sd {spread[1.0]:.4f} -> {spread[0.01]:.4f}")
