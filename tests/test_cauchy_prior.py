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


def test_at_MATCHED_kappa_recovery_the_heavy_TAIL_costs_the_effects_less():
    """THE CLAIM. A Gaussian tight enough to hand the nulls' exposure slope to κ
    also attenuates the real effects, because one scale does both jobs; that
    coupling is what refuted β_prior_σ = 0.1 on the benchmark. The Cauchy's
    penalty gradient 2β/(γ² + β²) is per-coefficient, so in principle it buys the
    same κ while leaving a large β alone.

    Compared at MATCHED κ rather than at a matched scale: the scale is a free knob,
    so the question is whether the frontier moves.
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

    # Match on κ: for each Gaussian setting take the Cauchy that recovered at least
    # as much of the drift (κ at least as negative), then compare attenuation.
    compared, wins = 0, 0
    for g in gauss:
        cands = [c for c in cauchy if c["κ"] <= g["κ"]]
        if not cands:
            continue
        c = max(cands, key=lambda v: v["κ"])      # the least aggressive match
        compared += 1
        wins += c["keep"] >= g["keep"]
    assert compared >= 2, f"no κ-matched pairs: {gauss} {cauchy}"
    # Not "every pair", because κ cannot be matched exactly on a 4-point grid; the
    # claim is that the family does not LOSE on the frontier.
    assert wins >= compared - 1, (
        f"cauchy attenuates more than the gaussian at matched kappa: "
        f"{gauss} vs {cauchy}")


def test_the_interval_prior_curvature_follows_the_family():
    """posterior_sd adds the prior's curvature. Under a Cauchy that is
    per-coefficient -- 2/(gamma^2 + beta^2) -- so a large coefficient gets a
    likelihood-driven interval while a null one stays prior-dominated. A scalar
    1/gamma^2 would hand the effects the null's width."""
    adata, beta, idx = make_data()
    nulls = beta == 0
    reg, est = fit(adata, "cauchy", 0.1)
    col = list(reg.design.design_info.column_names).index("exposure")
    sd = reg.posterior_sd()[col]
    assert sd[idx].mean() > sd[nulls].mean(), (
        f"effect sd {sd[idx].mean():.4f} not wider than null sd "
        f"{sd[nulls].mean():.4f}")
    # And the null width must not exceed what a Gaussian at the same scale gives,
    # since 2/gamma^2 > 1/sigma^2 at beta = 0.
    reg_g, _ = fit(adata, "normal", 0.1)
    sd_g = reg_g.posterior_sd()[col]
    assert sd[nulls].mean() <= sd_g[nulls].mean() * 1.05
