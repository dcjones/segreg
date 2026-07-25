"""Correctness of the contamination arms in the two-stream likelihoods.

Checks, against scipy, that (a) each contamination family really has the mean and
variance it is asked for, (b) the convolution likelihood equals a brute-force
convolution of the two component pmfs, (c) the "poisson" family is bit-for-bit the
old nb_conv behaviour, and (d) nb_mm matches the exact convolution's first two
moments.
"""

import numpy as np
import pytest
import torch
from scipy.stats import binom, nbinom, poisson

from segreg.losses import (
    _binom_n_p,
    _nb_contam_r,
    nb_loss_sparse,
    nbconv_loss_sparse,
    nbmm_loss_sparse,
)


def _csr(x):
    return torch.tensor(x, dtype=torch.float32).to_sparse_csr()


def _contam_pmf_np(delta, var, family, jmax=400):
    """Reference pmf of the contamination count, built with scipy from the same
    parameters the loss derives."""
    d = torch.tensor([delta], dtype=torch.float64)
    v = torch.tensor([var], dtype=torch.float64)
    j = np.arange(jmax + 1)
    if family == "poisson":
        return poisson.pmf(j, delta)
    if family == "binomial":
        n, p = _binom_n_p(d, v, 1e-12)
        return binom.pmf(j, float(n), float(p))
    r = float(_nb_contam_r(d, v, 1e-12))
    # scipy's NB is in (r, prob) form; mean = r(1-p)/p => p = r/(r+delta)
    return nbinom.pmf(j, r, r / (r + delta))


@pytest.mark.parametrize(
    "delta,var,family",
    [
        (2.0, 2.0, "poisson"),
        (2.0, 1.0, "binomial"),
        (0.7, 0.35, "binomial"),
        (5.0, 4.9, "binomial"),
        (2.0, 3.0, "nb"),
        (0.7, 1.4, "nb"),
    ],
)
def test_contam_moments(delta, var, family):
    """Each family hits the requested mean, and its variance matches the request
    (binomial errs high by at most one integer step in n, by construction)."""
    pmf = _contam_pmf_np(delta, var, family)
    j = np.arange(len(pmf))
    assert pmf.sum() == pytest.approx(1.0, abs=1e-9)
    mean = (j * pmf).sum()
    v = ((j - mean) ** 2 * pmf).sum()
    assert mean == pytest.approx(delta, rel=1e-6)
    if family == "binomial":
        # n is an integer, so the achievable variances are granular -- coarsely so
        # when delta < 1 and n is 1 or 2. Rounding n UP always lands on the
        # conservative side of the requested variance (wider, i.e. never claiming
        # more certainty than proseg reported), bounded by the Poisson variance.
        assert var - 1e-9 <= v <= delta + 1e-9
    else:
        assert v == pytest.approx(var, rel=1e-5)


@pytest.mark.parametrize("family", ["poisson", "binomial", "nb"])
def test_nbconv_matches_scipy_convolution(family):
    """The chunked logsumexp convolution equals a brute-force numpy convolution of
    NB(lam, r) with the contamination pmf, summed over all cells x genes."""
    rng = np.random.default_rng(0)
    n_cells, n_genes = 6, 4
    x = rng.poisson(3.0, size=(n_cells, n_genes)).astype(np.float32)
    lam = rng.uniform(0.5, 4.0, size=(n_cells, n_genes)).astype(np.float32)
    delta = rng.uniform(0.2, 3.0, size=(n_cells, n_genes)).astype(np.float32)
    if family == "poisson":
        var = delta.copy()
    elif family == "binomial":
        var = delta * rng.uniform(0.2, 0.9, size=delta.shape).astype(np.float32)
    else:
        var = delta * rng.uniform(1.5, 3.0, size=delta.shape).astype(np.float32)
    log_r = np.log(rng.uniform(1.0, 20.0, size=n_genes)).astype(np.float32)

    got = nbconv_loss_sparse(
        _csr(x),
        torch.tensor(lam),
        torch.tensor(delta),
        torch.tensor(log_r),
        conv_max=200,
        contam_var=torch.tensor(var),
        contam_family=family,
    ).item()

    r = np.exp(log_r)
    total = 0.0
    for c in range(n_cells):
        for g in range(n_genes):
            k = np.arange(int(x[c, g]) + 1)
            # P(X = x) = sum_k NB(k; lam) * P(C = x - k)
            nb_pmf = nbinom.pmf(k, r[g], r[g] / (r[g] + lam[c, g]))
            cpmf = _contam_pmf_np(float(delta[c, g]), float(var[c, g]), family)
            total += np.log(np.sum(nb_pmf * cpmf[int(x[c, g]) - k]))
    expected = -total / n_cells
    assert got == pytest.approx(expected, rel=1e-4)


def test_poisson_family_is_unchanged_default():
    """contam_var=None (the pre-existing call signature) and an explicit Poisson
    family with var == delta give identical values."""
    rng = np.random.default_rng(1)
    x = rng.poisson(2.0, size=(5, 3)).astype(np.float32)
    lam = torch.tensor(rng.uniform(0.5, 3.0, size=(5, 3)).astype(np.float32))
    delta = torch.tensor(rng.uniform(0.1, 2.0, size=(5, 3)).astype(np.float32))
    log_r = torch.tensor(np.log(rng.uniform(1.0, 10.0, size=3)).astype(np.float32))
    a = nbconv_loss_sparse(_csr(x), lam, delta, log_r, conv_max=100).item()
    b = nbconv_loss_sparse(
        _csr(x), lam, delta, log_r, conv_max=100,
        contam_var=delta, contam_family="poisson",
    ).item()
    assert a == b


def test_nbconv_without_delta_reduces_to_nb():
    rng = np.random.default_rng(2)
    x = rng.poisson(2.0, size=(5, 3)).astype(np.float32)
    lam = torch.tensor(rng.uniform(0.5, 3.0, size=(5, 3)).astype(np.float32))
    log_r = torch.tensor(np.log(rng.uniform(1.0, 10.0, size=3)).astype(np.float32))
    a = nbconv_loss_sparse(_csr(x), lam, None, log_r, conv_max=64).item()
    b = nb_loss_sparse(_csr(x), lam, log_r).item()
    assert a == pytest.approx(b, rel=1e-5)


@pytest.mark.parametrize("family,vscale", [("poisson", 1.0), ("nb", 2.0)])
def test_nbmm_matches_exact_moments(family, vscale):
    """nb_mm's NB has the same mean and variance as the exact two-stream model,
    wherever the matched variance stays above the mean (no clamping)."""
    lam, delta, r = 4.0, 2.0, 8.0
    var = delta * vscale
    x = np.arange(0, 400)
    log_r = torch.tensor([np.log(r)], dtype=torch.float32)
    lam_t = torch.tensor([[lam]], dtype=torch.float32)
    d_t = torch.tensor([[delta]], dtype=torch.float32)
    v_t = torch.tensor([[var]], dtype=torch.float32)

    # -loss * n_cells is the summed log-pmf; evaluate it at every x to get the pmf.
    logp = np.array(
        [
            -nbmm_loss_sparse(
                _csr(np.array([[xi]], dtype=np.float32)),
                lam_t, d_t, log_r, contam_var=v_t, contam_family=family,
            ).item()
            for xi in x
        ]
    )
    pmf = np.exp(logp)
    assert pmf.sum() == pytest.approx(1.0, rel=1e-4)
    mean = (x * pmf).sum()
    v = ((x - mean) ** 2 * pmf).sum()
    assert mean == pytest.approx(lam + delta, rel=1e-3)
    assert v == pytest.approx(lam + lam**2 / r + var, rel=1e-3)


def test_nbmm_floors_underdispersion_at_poisson():
    """A sub-Poisson contamination arm can push the matched variance below the mean;
    nb_mm cannot represent that and must floor at (near-)Poisson, not go invalid."""
    lam, delta, var, r = 0.05, 3.0, 0.5, 1000.0
    x = np.arange(0, 60)
    lam_t = torch.tensor([[lam]], dtype=torch.float32)
    d_t = torch.tensor([[delta]], dtype=torch.float32)
    v_t = torch.tensor([[var]], dtype=torch.float32)
    log_r = torch.tensor([np.log(r)], dtype=torch.float32)
    pmf = np.array(
        [
            np.exp(
                -nbmm_loss_sparse(
                    _csr(np.array([[xi]], dtype=np.float32)),
                    lam_t, d_t, log_r,
                    contam_var=v_t, contam_family="binomial",
                ).item()
            )
            for xi in x
        ]
    )
    assert np.isfinite(pmf).all()
    assert pmf.sum() == pytest.approx(1.0, rel=1e-3)
    mean = (x * pmf).sum()
    v = ((x - mean) ** 2 * pmf).sum()
    assert mean == pytest.approx(lam + delta, rel=1e-3)
    # true matched variance would be lam + lam^2/r + var = 0.55 < mean; the NB
    # floors at Poisson instead.
    assert v == pytest.approx(mean, rel=1e-2)
