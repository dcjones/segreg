"""Propagating κ's uncertainty into β's interval.

Reported β is what is left after δ = κ*u, so β's interval ought to include not
knowing the split. By default it does not -- posterior_sd conditions on the fitted
point κ. Two knobs answer that, and they are different in kind: `propagate_κ`
marginalizes κ out of the joint Laplace (no assumption, size set by the panel), and
`κ_rel_sd` adds a declared misspecification term (an assumption, mirroring the
post-hoc arm's kappa_rel_sd).

The first test is the one that matters: in a case with a closed form, does the Schur
complement return the right number?
"""

import numpy as np
import pytest

from segreg.regression import Regression
from test_drift import KAPPA_TRUE, make_data


def fit(adata, drift, propagate=False, rel_sd=None, prior_sigma=1.0,
        prior="normal", nepochs=600, seed=0):
    reg = Regression(adata, "~ 1 + exposure", include_mixing=True,
                     include_drift=True, drift=drift,
                     propagate_κ=propagate, κ_rel_sd=rel_sd)
    reg.fit(nepochs=nepochs, batch_size=300, seed=seed, verbose=False,
            compile=False, β_prior=prior, β_prior_σ=prior_sigma)
    return reg


def exposure_sd(reg):
    col = list(reg.design.design_info.column_names).index("exposure")
    return reg.posterior_sd()[col]


def test_the_marginal_kappa_sd_matches_its_CLOSED_FORM():
    """The real test of the linear algebra.

    The likelihood is EXACTLY flat along (β + c·u, κ − c), so along that direction
    the only curvature is what β's prior supplies: Σ_g u[k,g]²/σ² for a Gaussian.
    So κ's marginal sd must come out at σ/sqrt(Σ_g u²) -- a number computed from the
    covariate alone, with no reference to the Schur complement being tested.

    This is also the statement of DESIGN_v2 §62's "κ is outnumbered ~n_genes to 1":
    the precision is a SUM over genes, so κ's uncertainty falls like 1/sqrt(n_genes).
    """
    adata, drift, beta, u, _ = make_data()
    for σ in (1.0, 0.3):
        reg = fit(adata, drift, propagate=True, prior_sigma=σ)
        reg.posterior_sd()
        got = reg.κ_posterior_sd()[
            list(reg.design.design_info.column_names).index("exposure")]
        want = σ / np.sqrt((u**2).sum())
        assert got == pytest.approx(want, rel=0.15), (
            f"σ={σ}: marginal sd(κ) {got:.4f} against the closed form {want:.4f}")


def test_propagating_kappa_WIDENS_beta_and_widens_it_where_u_is_LARGE():
    """The correction is rank-ncov and enters through u, so the extra variance must
    scale with u² rather than being a uniform inflation -- a gene whose foreign
    share does not respond to exposure has nothing extra to be uncertain about."""
    adata, drift, beta, u, _ = make_data()
    sd0 = exposure_sd(fit(adata, drift))
    sd1 = exposure_sd(fit(adata, drift, propagate=True))
    assert (sd1 >= sd0 - 1e-12).all(), "marginalizing κ NARROWED an interval"
    assert sd1.mean() > sd0.mean(), "no widening at all"
    extra = sd1**2 - sd0**2
    assert np.corrcoef(extra, u**2)[0, 1] > 0.9, (
        f"extra variance does not track u²: corr {np.corrcoef(extra, u**2)[0,1]:.2f}")


def test_a_PINNED_kappa_gets_no_correction():
    """κ_prior_σ → 0 is the fixed-κ arm: nothing is being estimated, so there is no
    uncertainty to propagate. The Schur complement has to reproduce that limit."""
    adata, drift, beta, u, _ = make_data()
    reg_free = Regression(adata, "~ 1 + exposure", include_mixing=True,
                          include_drift=True, drift=drift, propagate_κ=True)
    reg_free.fit(nepochs=600, batch_size=300, seed=0, verbose=False, compile=False,
                 β_prior="normal")
    reg_pin = Regression(adata, "~ 1 + exposure", include_mixing=True,
                         include_drift=True, drift=drift, propagate_κ=True)
    reg_pin.fit(nepochs=600, batch_size=300, seed=0, verbose=False, compile=False,
                β_prior="normal", κ_prior_μ=KAPPA_TRUE, κ_prior_σ=1e-3)
    col = list(reg_free.design.design_info.column_names).index("exposure")
    reg_free.posterior_sd(); reg_pin.posterior_sd()
    assert reg_pin.κ_posterior_sd()[col] < 0.1 * reg_free.κ_posterior_sd()[col]


def test_the_declared_term_is_exactly_what_it_says():
    """κ_rel_sd is an assumption, not an estimate, so it should be trivially
    auditable: exactly (rel·|κ|·|u|)² added in quadrature."""
    adata, drift, beta, u, _ = make_data()
    reg0 = fit(adata, drift)
    reg1 = fit(adata, drift, rel_sd=0.5)
    col = list(reg0.design.design_info.column_names).index("exposure")
    κ = float(reg0.get_drift_scales()["exposure"])
    want = np.sqrt(reg0.posterior_sd()[col]**2 + (0.5 * abs(κ) * np.abs(u))**2)
    assert reg1.posterior_sd()[col] == pytest.approx(want, rel=1e-6)


def test_it_refuses_to_compose_with_the_sandwich():
    adata, drift, _, _, _ = make_data()
    reg = Regression(adata, "~ 1 + exposure", include_mixing=True,
                     include_drift=True, drift=drift, propagate_κ=True,
                     mixing_interval_correction=True)
    reg.fit(nepochs=50, batch_size=300, seed=0, verbose=False, compile=False)
    with pytest.raises(ValueError, match="cannot be combined"):
        reg.posterior_sd()
