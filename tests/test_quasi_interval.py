"""The quasi-Poisson interval: does swapping the variance function report the sd the
estimator actually has?

WHY. `posterior_sd` inverts the NB Fisher information, whose per-cell contribution
r/(mu(r+mu)) SATURATES at r as mu grows -- so on a well-measured gene the reported
width stops responding to the data. Graded against the true sampling sd on
simple-seg-sim (measured across emission replicates, so it references no truth), the
NB interval is 1.55-1.95x too wide on exactly those genes, and that width is the whole
power deficit against a plain quasi-Poisson GLM. DESIGN_v2 §84-§85.

These tests use POISSON data, which is the case where the answer is known: phi is 1,
the honest Fisher information is diag(mu), and the closed form is computable in numpy
without reference to the code path being tested. The last test uses genuinely
overdispersed data to check the flag is not just "always narrower" -- phi has to FIND
the overdispersion when it is there.
"""

import numpy as np
import pytest
from anndata import AnnData
from scipy.sparse import csr_matrix, diags

from segreg.regression import Regression

NCELLS, NGENES = 1200, 40


def make_poisson(seed=0, r=None, depth=1.5):
    """Counts from a log-linear Poisson model, or NB with dispersion `r`.

    One covariate and an intercept, identity mixing, so mu is the cell's own rate
    and the design is small enough for the 2x2 closed form below.

    `depth` is mean counts per cell per gene, and it is NOT a free choice: phi is a
    Pearson dispersion, so any relative error in the fitted mean enters it as
    (err*mu)^2/mu = err^2 * mu -- i.e. mean-model misfit reads as dispersion, and it
    does so IN PROPORTION TO DEPTH. Measured here: at depth 57 the fitted mu is 7.6%
    off per cell (the offset is the observed total, which is itself noisy) and phi
    comes out at 2.43 against 1.00 computed at the true lambda. At depth 1.5 -- the
    Xenium regime these panels actually live in, 0.2-3 counts/cell/gene on both
    benchmark datasets -- the same misfit is worth <1% and phi measures dispersion.
    `test_phi_absorbs_mean_misfit_IN_PROPORTION_TO_DEPTH` pins that down, because it
    is the condition under which this whole option is trustworthy.
    """
    rng = np.random.default_rng(seed)
    exposure = rng.normal(0.0, 1.0, NCELLS)
    log_size = np.log(rng.integers(1500, 2500, NCELLS).astype(float))
    beta = rng.choice([-0.6, 0.0, 0.6], NGENES)
    gene_bias = np.full(NGENES, np.log(depth) - np.log(2000.0))
    lam = np.exp(exposure[:, None] * beta[None, :]
                 + gene_bias[None, :] + log_size[:, None])
    if r is None:
        X = rng.poisson(lam)
    else:
        # NB as a gamma-Poisson mixture, mean lam and variance lam + lam^2/r.
        X = rng.poisson(rng.gamma(shape=r, scale=lam / r))
    X = X.astype(np.float32)
    adata = AnnData(X=csr_matrix(X))
    adata.var_names = [f"g{j}" for j in range(NGENES)]
    adata.obs["exposure"] = exposure
    adata.obsp["state_transitions"] = csr_matrix(
        diags(np.asarray(X.sum(axis=1)).ravel()))
    adata.obs["to_bg_trans_count"] = np.zeros(NCELLS)
    return adata, beta, exposure, log_size


def make_quasi_shaped(seed=0, c=0.25, depth=1.5):
    """Counts whose variance is LINEAR in the mean: Var = lam*(1 + 1/c).

    A gamma-Poisson mixture with the gamma's shape proportional to lam, so the
    overdispersion factor is constant across cells instead of growing with the mean.
    This is the shape DESIGN_v2 §85 measures on both benchmark datasets -- observed
    squared residuals at 53-74% of what the fitted NB claims, and an NB variance
    function that therefore misprices the well-measured cells. It is the case the
    quasi weight exists for, and the honest slope sd here is sqrt(phi) times the
    Poisson one, which makes it checkable.
    """
    rng = np.random.default_rng(seed)
    exposure = rng.normal(0.0, 1.0, NCELLS)
    log_size = np.log(rng.integers(1500, 2500, NCELLS).astype(float))
    beta = rng.choice([-0.6, 0.0, 0.6], NGENES)
    gene_bias = np.full(NGENES, np.log(depth) - np.log(2000.0))
    lam = np.exp(exposure[:, None] * beta[None, :]
                 + gene_bias[None, :] + log_size[:, None])
    X = rng.poisson(rng.gamma(shape=c * lam, scale=1.0 / c)).astype(np.float32)
    adata = AnnData(X=csr_matrix(X))
    adata.var_names = [f"g{j}" for j in range(NGENES)]
    adata.obs["exposure"] = exposure
    adata.obsp["state_transitions"] = csr_matrix(
        diags(np.asarray(X.sum(axis=1)).ravel()))
    adata.obs["to_bg_trans_count"] = np.zeros(NCELLS)
    return adata, beta, 1.0 + 1.0 / c


def fit(adata, quasi=False, nepochs=1000, seed=0, prior_sigma=1.0):
    reg = Regression(adata, "~ 1 + exposure", include_mixing=True,
                     quasi_interval=quasi)
    reg.fit(nepochs=nepochs, batch_size=400, seed=seed, verbose=False,
            compile=False, β_prior="normal", β_prior_σ=prior_sigma)
    return reg


def expo_col(reg):
    return list(reg.design.design_info.column_names).index("exposure")


def closed_form_poisson_sd(reg, prior_sigma=1.0):
    """sqrt(diag((X' diag(mu) X + I/sigma^2)^-1)) for the exposure column, per gene.

    Built in numpy from the FITTED mean, so it tests the weight and the linear
    algebra in posterior_sd without reusing either.
    """
    m = reg.model
    B = m.β_μ.detach().cpu().numpy().astype(np.float64)
    gb = m.gene_bias.detach().cpu().numpy().astype(np.float64)
    X = np.asarray(reg.design, dtype=np.float64)
    ls = reg.log_size.astype(np.float64)
    μ = np.exp(X @ B + gb[None, :] + ls[:, None])
    j = expo_col(reg)
    out = np.empty(B.shape[1])
    for g in range(B.shape[1]):
        H = X.T @ (X * μ[:, [g]]) + np.eye(X.shape[1]) / prior_sigma ** 2
        out[g] = np.sqrt(np.linalg.inv(H)[j, j])
    return out


def test_it_is_OFF_by_default_and_bit_identical():
    """The flag must not change an existing arm's numbers by existing."""
    adata, _, _, _ = make_poisson()
    reg = fit(adata, quasi=False)
    a = reg.posterior_sd()
    b = reg.posterior_sd(quasi=False)
    assert np.array_equal(a, b)
    assert np.all(np.isnan(reg.interval_phi)), (
        "interval_phi must be NaN when the quasi weight was not used, so a caller "
        "cannot record an NB fit's dispersion as if it were phi")


def test_on_POISSON_data_the_quasi_sd_matches_the_CLOSED_FORM():
    """The real test of the weight: phi is ~1 here, so 1/(phi*mu) must reproduce the
    Poisson Fisher information exactly, computed independently in numpy."""
    adata, _, _, _ = make_poisson()
    reg = fit(adata, quasi=True)
    got = reg.posterior_sd()[expo_col(reg)]
    want = closed_form_poisson_sd(reg)
    # phi is estimated, not pinned at 1, so it carries sampling noise of order
    # sqrt(2/ncells) ~ 4% -- and it is floored at 1, which biases it up a little.
    assert np.allclose(got, want, rtol=0.12), (
        f"median ratio {np.median(got / want):.4f}, "
        f"max |ratio-1| {np.abs(got / want - 1).max():.4f}")
    assert np.median(reg.interval_phi) == pytest.approx(1.0, abs=0.15), (
        f"phi median {np.median(reg.interval_phi):.3f} on Poisson data at this "
        "depth -- see make_poisson's docstring on depth and mean misfit")


def test_on_SELF_CONSISTENT_data_the_two_weights_AGREE():
    """The control, and the honest limit of this option.

    On data generated from the model the NB weight is CORRECT, so the quasi one has
    nothing to win: r runs away on Poisson data (measured r_hat ~33, phi ~1.01) and
    recovers the truth on NB data (r_hat 2.07 against r=2), and the two reported sds
    agree to a few percent. Anyone reading the quasi weight as uniformly narrower is
    reading it wrong -- it only differs where the NB variance FUNCTION is wrong.

    On NB data it does come out slightly NARROWER (~1.16x), which is the direction to
    remember: under a correctly specified NB, a single phi per gene applies the average
    inflation to every cell instead of the mean-dependent one, and that understates.
    """
    adata, _, _, _ = make_poisson()                       # Poisson: nothing to fix
    nb, qp = fit(adata, quasi=False), fit(adata, quasi=True)
    j = expo_col(nb)
    pois = closed_form_poisson_sd(nb)
    assert np.median(nb.posterior_sd()[j] / pois) == pytest.approx(1.0, abs=0.10)
    assert np.median(qp.posterior_sd()[j] / pois) == pytest.approx(1.0, abs=0.10)

    adata, _, _, _ = make_poisson(r=2.0)                  # NB: NB weight is right
    nb, qp = fit(adata, quasi=False), fit(adata, quasi=True)
    ratio = np.median(nb.posterior_sd()[j] / qp.posterior_sd()[j])
    assert 1.0 < ratio < 1.4, (
        f"sd_nb/sd_qp {ratio:.3f} on correctly-specified NB data; the quasi weight "
        "is expected to be mildly anticonservative there, not wildly off")


def test_phi_recovers_the_dispersion_a_CLOSED_FORM_predicts():
    """phi is an estimate of a known quantity here: for NB(r) at mean mu the Pearson
    dispersion is 1 + mu/r, so at depth 1.5 and r=2 it must come out near 1.75."""
    adata, _, _, _ = make_poisson(r=2.0, depth=1.5)
    reg = fit(adata, quasi=True)
    reg.posterior_sd()
    assert np.median(reg.interval_phi) == pytest.approx(1.0 + 1.5 / 2.0, rel=0.15), (
        f"phi median {np.median(reg.interval_phi):.3f} against 1 + mu/r = 1.75")


def test_WHERE_THE_NB_SHAPE_IS_WRONG_the_quasi_weight_is_right_and_NB_is_wide():
    """The case the option exists for, with a checkable answer.

    Variance linear in the mean, so the honest slope sd is sqrt(phi) x the Poisson
    one. The NB weight must approximate a linear variance with mu + mu^2/r, which
    forces r small (measured r_hat 0.46 at phi 5, the same regime as the real data's
    0.07) and misprices the high-mu cells -- the mechanism DESIGN_v2 §85 identifies.
    """
    for c, want_nb_wide in ((0.25, 1.10), (1.0, 1.08)):
        adata, _, phi_true = make_quasi_shaped(c=c)
        nb, qp = fit(adata, quasi=False), fit(adata, quasi=True)
        j = expo_col(nb)
        want = np.sqrt(phi_true) * closed_form_poisson_sd(nb)
        got_nb = np.median(nb.posterior_sd()[j] / want)
        got_qp = np.median(qp.posterior_sd()[j] / want)
        assert got_nb > want_nb_wide, (
            f"c={c}: the NB weight should be too wide here; got {got_nb:.3f}")
        assert got_qp == pytest.approx(1.0, abs=0.08), (
            f"c={c}: quasi sd / honest sd {got_qp:.3f}")
        assert got_nb > got_qp


def test_phi_absorbs_mean_misfit_IN_PROPORTION_TO_DEPTH():
    """The condition under which the quasi weight is trustworthy, made a test.

    phi cannot tell dispersion from a mean that is slightly wrong: a relative error
    `e` in mu contributes e^2 * mu to the Pearson dispersion. So on DEEP data a
    well-fitted model still reports phi >> 1 and the interval is conservative, while
    at Xenium depth the same misfit is invisible. Both directions are checked, because
    the failure this guards against is someone reusing the option on deep counts and
    reading the widening as real overdispersion.

    Poisson data both times, so the honest phi is exactly 1.0 by construction.
    """
    shallow, deep = {}, {}
    for depth, store in ((1.5, shallow), (57.0, deep)):
        adata, _, _, _ = make_poisson(depth=depth)
        reg = fit(adata, quasi=True)
        reg.posterior_sd()
        store["phi"] = float(np.median(reg.interval_phi))
    assert shallow["phi"] == pytest.approx(1.0, abs=0.15), shallow
    assert deep["phi"] > 1.5, (
        f"expected phi to inflate with depth on Poisson data; got {deep['phi']:.3f}. "
        "If this stopped happening the fitted mean got much more accurate, which is "
        "good news -- re-derive the caveat rather than deleting the test.")


def test_it_REFUSES_to_combine_with_the_sandwich():
    adata, _, _, _ = make_poisson()
    reg = fit(adata, quasi=True)
    with pytest.raises(ValueError, match="double-count"):
        reg.posterior_sd(sandwich=True)
