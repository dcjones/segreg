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
import pytest
from anndata import AnnData
from scipy.sparse import csr_matrix, diags

from segreg.regression import Regression

NCELLS, NGENES = 900, 60
N_EFFECT = 6
EFFECT = 0.8


def make_data(seed=0):
    """Sparse exposure coefficients, identity mixing."""
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


def test_an_unknown_prior_family_is_an_error_not_a_silent_gaussian():
    adata, _, _ = make_data()
    with pytest.raises(ValueError, match="must be 'normal' or 'cauchy'"):
        fit(adata, prior="laplace", nepochs=1)


def test_the_family_barely_moves_the_POINT_estimates():
    """With β facing the likelihood alone, the prior is swamped: over σ ∈ {1.0, 0.3}
    the null |bias| moves in the 4th decimal (0.0401 -> 0.0400) and swapping the
    family moves the coefficients by ~3e-4. So a Cauchy is NOT a general-purpose
    tightening of the point estimates; where it acts is the interval (below).
    """
    adata, beta, _ = make_data()
    nulls = beta == 0
    _, g = fit(adata, "normal", 0.3)
    _, c = fit(adata, "cauchy", 0.3)
    assert np.abs(g[nulls]).mean() == pytest.approx(
        np.abs(c[nulls]).mean(), abs=0.01)


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
