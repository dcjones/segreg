"""The long-range leak nuisance.

lambda_obs[i, g] += sum_s rho_s E[i, s] P[s, g], with rho shared across genes. The
scenario is the one it exists for: receiver cells next to a leaky source pick up
the source's profile in proportion to their exposure, and the regression's niche
coefficient (receiver expression vs. exposure) is spurious. The bar: default
unchanged, a known rate removes the spurious slope, a free rate recovers it from
the data (because the leak is the source's whole profile, in proportion), a real
gene-specific effect survives, and misalignment is an error.
"""

import numpy as np
import pytest
from anndata import AnnData
from scipy.sparse import csr_matrix, diags

from segreg.regression import Regression

NCELLS, NGENES = 1500, 60
RHO = 0.4


def make(seed=0, rho=RHO, planted=0.0):
    """Receivers with a native profile; exposure E to one leaky source with
    profile P. Gene 0 additionally carries a REAL niche effect `planted` on the
    log scale (receiver induction that is not source-shaped)."""
    rng = np.random.default_rng(seed)
    size = rng.integers(150, 250, NCELLS).astype(float)
    native = rng.dirichlet(np.full(NGENES, 2.0))
    src = rng.dirichlet(np.full(NGENES, 0.3))  # a distinct, peaky source profile
    E = rng.gamma(1.0, 100.0, NCELLS)  # expected leak count into the cell at rho = 1
    z = (E - E.mean()) / E.std()  # the niche covariate a regression would use
    eff = np.zeros(NGENES)
    eff[0] = planted
    lam = size[:, None] * native[None, :] * np.exp(z[:, None] * eff[None, :])
    lam = lam + rho * E[:, None] * src[None, :]
    X = rng.poisson(lam).astype(np.float32)
    adata = AnnData(X=csr_matrix(X))
    adata.var_names = [f"g{j}" for j in range(NGENES)]
    adata.obs["z"] = z
    adata.uns["native"] = native
    adata.obsp["state_transitions"] = csr_matrix(diags(np.asarray(X.sum(1)).ravel()))
    adata.obs["to_bg_trans_count"] = np.zeros(NCELLS)
    leak = dict(E=E[:, None], P=src[None, :], genes=list(adata.var_names), names=["src"])
    return adata, leak, src


def native_of(adata):
    return np.asarray(adata.uns["native"])


def fit(adata, leak=None, offset_rho=None, **kw):
    if leak is not None and offset_rho is not None:
        leak = dict(leak, offset_rho=offset_rho)
    r = Regression(adata, "~ 1 + z", include_mixing=False, leak=leak)
    r.fit(nepochs=600, seed=0, device="cpu", verbose=False, compile=False, **kw)
    β = r.get_regression_coefficients().query("Covariate == 'z'").set_index("Gene")["Mean"]
    return r, β.reindex(adata.var_names).to_numpy()


def test_default_is_UNCHANGED():
    adata, _, _ = make()
    r = Regression(adata, "~ 1 + z", include_mixing=False)
    assert r.leak is None


def test_leak_makes_SPURIOUS_slopes_and_a_known_rate_removes_them():
    adata, leak, src = make()
    top = np.argsort(src / native_of(adata))[-5:]  # genes the source over-represents
    _, β0 = fit(adata)
    assert β0[top].mean() > 0.2, "the scenario should produce spurious positive slopes"
    # The offset must be the NATIVE size, or the known rate double-counts the leak.
    _, β1 = fit(adata, leak, offset_rho=RHO, leak_prior_μ=np.log(RHO), leak_prior_σ=0.0)
    assert abs(β1[top].mean()) < 0.3 * β0[top].mean()


def test_a_FREE_rate_is_recovered_because_it_is_shared_across_genes():
    adata, leak, src = make()
    r, β = fit(adata, leak, leak_prior_σ=None)
    rho = float(r.get_leak_rates()["src"])
    assert 0.5 * RHO < rho < 1.5 * RHO
    top = np.argsort(src / native_of(adata))[-5:]
    _, β0 = fit(adata)
    assert abs(β[top].mean()) < 0.3 * β0[top].mean()


def test_a_REAL_gene_specific_effect_survives_the_leak_term():
    adata, leak, src = make(planted=0.5)
    assert np.argsort(src)[-1] != 0 or src[0] < 0.2  # gene 0 is not the source's main gene
    _, β = fit(adata, leak, leak_prior_σ=None)
    assert β[0] > 0.3


def test_offset_rho_gives_the_NATIVE_size():
    adata, leak, _ = make()
    r = Regression(adata, "~ 1 + z", leak=dict(leak, offset_rho=RHO))
    total = np.asarray(adata.X.sum(1)).ravel()
    want = np.log(np.maximum(total - RHO * leak["E"][:, 0] * leak["P"].sum(), 1.0))
    assert np.allclose(r.log_size, want, atol=1e-4)


def test_misaligned_leak_is_an_ERROR():
    adata, leak, _ = make()
    with pytest.raises(ValueError):
        Regression(adata, "~ 1 + z", leak=dict(leak, E=leak["E"][:-1]))
    with pytest.raises(ValueError):
        Regression(adata, "~ 1 + z", leak=dict(leak, genes=[f"x{j}" for j in range(NGENES)]))


def test_declared_rate_uncertainty_WIDENS_only_the_source_shaped_genes():
    """With rho pinned at 0 for the point estimate and a declared sd on it, the
    interval should widen for the genes leak could explain (the source's
    over-represented genes) and stay put for the rest."""
    adata, leak, src = make()
    ratio = src / native_of(adata)
    top, bottom = np.argsort(ratio)[-5:], np.argsort(ratio)[:5]

    def widths(isd):
        lk = dict(leak, interval_sd=isd)
        r = Regression(adata, "~ 1 + z", include_mixing=False, leak=lk)
        r.fit(nepochs=600, seed=0, device="cpu", verbose=False, compile=False,
              leak_prior_μ=-np.inf, leak_prior_σ=0.0)
        c = r.get_regression_coefficients(credible_interval=0.95).query("Covariate == 'z'")
        return (c["Upper"] - c["Lower"]).to_numpy()

    w0, w1 = widths(0.0), widths(RHO)
    assert (w1[top] / w0[top]).mean() > 2.0
    assert np.abs(w1[bottom] / w0[bottom] - 1).max() < 0.01
    # and the widening tracks how leak-shaped a gene is
    from scipy.stats import spearmanr
    assert spearmanr(ratio, w1 / w0).statistic > 0.8
