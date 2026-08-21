"""The Cauchy prior on beta: is it live, and does it decouple what a Gaussian couples?

The claim being tested is narrow. A Gaussian prior has ONE scale doing two jobs:
shrinking the (overwhelming majority) null coefficients toward zero and holding
back the handful of real effects. Tightening it does both -- which is what refuted
`beta_prior_sigma = 0.1` on the benchmark (simple-seg-sim DESIGN_v2 §62: nominal
coverage 0.20 because the intervals no longer covered anything). A Cauchy's
penalty gradient decays like 2/beta in the tail, so in principle it can do the
first job without the second.

Data are sparse by construction (6 real effects in 60 genes), which is the regime
the prior asserts. These are self-consistency tests: whether real panels are this
sparse is what bench/ measures.
"""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy.sparse import csr_matrix, diags

from segreg.regression import Regression

NCELLS, NGENES = 900, 60
N_EFFECT = 6
EFFECT = 0.8


def make_data(seed=0):
    """Sparse exposure coefficients, identity mixing, no drift."""
    rng = np.random.default_rng(seed)
    genes = [f"g{j}" for j in range(NGENES)]
    exposure = rng.normal(0.0, 1.0, NCELLS)
    # DELIBERATELY SHALLOW: ~3 counts per gene per cell. At panel-realistic depth
    # (33/gene) the likelihood swamps any prior on this design -- normal σ=1.0 and
    # σ=0.3 give null |bias| identical to 4 decimals -- so a deep synthetic tests
    # nothing about a prior. Breast's panel runs ~1-2 counts/gene/cell and the WTA
    # far less, which is the regime where the choice of family can matter at all.
    log_size = np.log(rng.integers(150, 250, NCELLS).astype(float))

    beta = np.zeros(NGENES)
    idx = rng.choice(NGENES, N_EFFECT, replace=False)
    beta[idx] = rng.choice([-EFFECT, EFFECT], N_EFFECT)

    gene_bias = np.full(NGENES, -np.log(NGENES))
    lam = np.exp(exposure[:, None] * beta[None, :]
                 + gene_bias[None, :] + log_size[:, None])
    X = rng.poisson(lam).astype(np.float32)

    adata = AnnData(X=csr_matrix(X))
    adata.var_names = genes
    adata.obs["exposure"] = exposure
    adata.obsp["state_transitions"] = csr_matrix(
        diags(np.asarray(X.sum(axis=1)).ravel()))
    adata.obs["to_bg_trans_count"] = np.zeros(NCELLS)
    return adata, beta, idx


def fit(adata, prior="normal", scale=1.0, seed=0, nepochs=1200):
    reg = Regression(adata, "~ 1 + exposure", include_mixing=True)
    reg.fit(nepochs=nepochs, batch_size=300, seed=seed, verbose=False,
            compile=False, β_prior=prior, β_prior_σ=scale)
    df = reg.get_regression_coefficients()
    est = (df[df["Covariate"] == "exposure"]
           .set_index("Gene")["Mean"].reindex(adata.var_names).to_numpy())
    return reg, est



def make_drift_data(seed=0):
    """The sparse-beta data of test_drift, whose observed slope is beta + kappa*u.

    Reused rather than rebuilt: the drift term is the only place a prior on beta
    decides anything (see the test below), so the family comparison has to be run
    on the same generative form the drift tests use.
    """
    from test_drift import make_data as _md
    return _md(seed)


def fit_drift(adata, drift, prior, scale, seed=0, nepochs=1200):
    reg = Regression(adata, "~ 1 + exposure", include_mixing=True,
                     include_drift=True, drift=drift)
    reg.fit(nepochs=nepochs, batch_size=300, seed=seed, verbose=False,
            compile=False, β_prior=prior, β_prior_σ=scale)
    df = reg.get_regression_coefficients()
    est = (df[df["Covariate"] == "exposure"]
           .set_index("Gene")["Mean"].reindex(adata.var_names).to_numpy())
    return reg, est


def kappa_of(reg):
    return float(reg.get_drift_scales()["exposure"])


def test_an_unknown_prior_family_is_an_error_not_a_silent_gaussian():
    adata, _, _ = make_data()
    with pytest.raises(ValueError, match="must be 'normal' or 'cauchy'"):
        fit(adata, prior="laplace", nepochs=1)


def test_WITHOUT_the_drift_term_neither_family_matters_and_that_is_the_point():
    """MEASURED, and it frames every test below.

    With β facing the likelihood alone, the prior is swamped: over σ ∈ {1.0, 0.3}
    the null |bias| moves in the 4th decimal (0.0401 -> 0.0400) and swapping the
    family moves the coefficients by ~3e-4. So a Cauchy is NOT a general-purpose
    tightening -- there is nothing here for it to tighten.

    Where a prior on β decides anything is the drift model, whose likelihood is
    EXACTLY invariant along (β + c*u, κ - c). The prior alone picks the split, and
    that is the only place a family swap can act.
    """
    adata, beta, _ = make_data()
    nulls = beta == 0
    _, g = fit(adata, "normal", 0.3)
    _, c = fit(adata, "cauchy", 0.3)
    assert np.abs(g[nulls]).mean() == pytest.approx(
        np.abs(c[nulls]).mean(), abs=0.01)


def test_the_prior_is_LIVE_in_the_split():
    """'no effect' and 'not wired up' look identical in a summary table, so the
    family swap has to be shown to MOVE the split before anything is read off it.

    Checked at a TIGHT scale, because at the default one it barely moves --
    measured κ -0.670 (normal σ=1) against -0.687 (cauchy γ=1). That is expected
    rather than a wiring failure: the Cauchy's curvature at zero is 2/γ², so
    γ=1 acts on the nulls like a Gaussian σ=0.71, and it is only where the
    Gaussian is tight enough to bite that the two families can disagree.
    """
    adata, drift, beta, u, _ = make_drift_data()
    kg = kappa_of(fit_drift(adata, drift, "normal", 0.1)[0])
    kc = kappa_of(fit_drift(adata, drift, "cauchy", 0.1)[0])
    assert abs(kc - kg) > 0.05, f"cauchy changed nothing: kappa {kg} -> {kc}"


def test_at_MATCHED_null_SHRINKAGE_the_heavy_TAIL_costs_the_effects_less():
    """THE CLAIM, and the measurement that qualifies it.

    A Gaussian tight enough to hand the nulls' exposure slope to κ also attenuates
    the real effects, because one scale does both jobs; that coupling is what
    refuted β_prior_σ = 0.1 on the benchmark (DESIGN_v2 §62). The Cauchy's penalty
    gradient 2β/(γ² + β²) is per-coefficient, so it can in principle shrink the
    nulls without the same tax.

    MEASURED here (κ_true = -1.5, `keep` = mean |β̂|/0.8 over the 6 real effects,
    `null` = mean |β̂| over the 54 nulls):

        gaussian σ   1.0     0.3     0.1     0.03
        κ           -0.670  -0.922  -1.345  -1.358
        keep         0.878   0.871   0.851   0.745
        null         0.228   0.159   0.108   0.095

        cauchy γ     1.0     0.3     0.1     0.03
        κ           -0.687  -0.861  -1.103  -1.168
        keep         0.878   0.874   0.868   0.867
        null         0.223   0.175   0.118   0.106

    Two things follow, and the second is the reason this is not a free win:

    * The decoupling is REAL but small. At matched null shrinkage (γ=0.03's 0.106
      against σ=0.1's 0.108) the Cauchy keeps 0.867 of the effects against 0.851,
      and it never falls off the cliff the Gaussian hits at σ=0.03 (0.745).
    * The Cauchy recovers LESS κ at every scale and saturates around -1.17,
      short of the truth. Its tail is exactly what lets a coefficient keep the
      exposure-correlated variation the drift term is trying to claim, so
      "attenuates the effects less" and "corrects less" are the same fact seen
      twice. Which of the two the benchmark rewards is not decidable here.

    Matched on NULL SHRINKAGE rather than on κ: κ cannot be matched at all at the
    tight end (no Cauchy reaches -1.345), and it is the nulls the prior is nominally
    there to shrink.
    """
    adata, drift, beta, u, idx = make_drift_data()

    def run(prior, scale):
        reg, est = fit_drift(adata, drift, prior, scale)
        return dict(scale=scale, κ=kappa_of(reg),
                    keep=float(np.abs(est[idx]).mean() / EFFECT),
                    null=float(np.abs(est[beta == 0]).mean()))

    grid = (1.0, 0.3, 0.1, 0.03)
    gauss = [run("normal", s) for s in grid]
    cauchy = [run("cauchy", s) for s in grid]
    print("\ngaussian", gauss, "\ncauchy  ", cauchy)

    # Both families must be monotone in their scale, or there is no frontier to
    # compare -- only noise.
    for fam, name in ((gauss, "gaussian"), (cauchy, "cauchy")):
        nulls = [v["null"] for v in fam]
        assert nulls == sorted(nulls, reverse=True), f"{name} not monotone: {fam}"

    # Frontier comparison: for each Gaussian setting, the Cauchy that shrinks the
    # nulls at least as hard (within a 0.005 tolerance, since the grids do not
    # line up exactly) must not attenuate the effects more.
    compared = 0
    for g in gauss:
        cands = [c for c in cauchy if c["null"] <= g["null"] + 0.005]
        if not cands:
            continue
        c = min(cands, key=lambda v: abs(v["null"] - g["null"]))
        compared += 1
        assert c["keep"] >= g["keep"] - 0.01, (
            f"cauchy γ={c['scale']} shrank the nulls to {c['null']:.3f} against "
            f"gaussian σ={g['scale']}'s {g['null']:.3f} but kept only "
            f"{c['keep']:.3f} of the effects against {g['keep']:.3f}")
    assert compared >= 2, f"no comparable pairs: {gauss} {cauchy}"

    # And the specific failure the Gaussian has: at its tightest setting it gives
    # up 13% of the effects for 0.013 of null bias. The Cauchy's tightest does not.
    assert cauchy[-1]["keep"] > gauss[-1]["keep"] + 0.05, (
        f"cauchy tail did not avoid the gaussian's collapse: "
        f"{cauchy[-1]} vs {gauss[-1]}")


def test_the_interval_prior_curvature_follows_the_family():
    """posterior_sd adds the prior's curvature, and under a Cauchy it is
    PER-COEFFICIENT: 2/(γ² + β̂²), so a null is prior-dominated while a large
    coefficient is handed back to the likelihood.

    Tested as a RATIO against the Gaussian at the same scale, not as
    "effects get wider intervals than nulls" -- they do not, and that was our
    first version of this test. The likelihood term dominates both, and a real
    effect raises λ where the exposure is high, so it is the BETTER-determined
    coefficient (sd 0.0156 against the nulls' 0.0176). What the family changes is
    the prior's share, which the ratio isolates: at β = 0 the Cauchy's 2/γ²
    exceeds the Gaussian's 1/γ², so nulls narrow, while at |β| >> γ its
    contribution vanishes.
    """
    adata, beta, idx = make_data()
    nulls = beta == 0
    scale = 0.1
    reg_c, _ = fit(adata, "cauchy", scale)
    reg_g, _ = fit(adata, "normal", scale)
    col = list(reg_c.design.design_info.column_names).index("exposure")
    sd_c, sd_g = reg_c.posterior_sd()[col], reg_g.posterior_sd()[col]
    ratio_eff = float((sd_c[idx] / sd_g[idx]).mean())
    ratio_null = float((sd_c[nulls] / sd_g[nulls]).mean())
    assert ratio_eff > ratio_null, (
        f"cauchy/gaussian width ratio {ratio_eff:.3f} at the effects is not above "
        f"{ratio_null:.3f} at the nulls -- the curvature is not per-coefficient")
    assert ratio_null <= 1.0 + 1e-6, f"nulls not narrowed: ratio {ratio_null:.3f}"
