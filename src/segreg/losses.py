import math

import torch
import torch.nn.functional as F


def poisson_loss(x_sub_tensor, mu):
    """Poisson negative log-likelihood, summed over genes, averaged over cells."""
    mu_f = mu.float()
    return (mu_f - x_sub_tensor * torch.log(mu_f.clamp(min=1e-8))).sum(dim=-1).mean()


def nb_loss(x_sub_tensor, mu, log_r):
    """Negative binomial reconstruction loss, summed over genes, averaged over cells.

    Parameterized by r = exp(log_r) (= 1 / psi_g in the paper's dispersion notation),
    so that Var[X] = mu + mu^2 / r = mu + psi_g * mu^2.
    """
    r = torch.exp(log_r).clamp(min=1e-3)
    mu_nb = mu.float()
    eps = 1e-8
    log_r_over_r_plus_mu = torch.log(r / (r + mu_nb + eps))
    log_mu_over_r_plus_mu = torch.log((mu_nb + eps) / (r + mu_nb + eps))
    return (
        -(
            torch.lgamma(x_sub_tensor + r)
            - torch.lgamma(r)
            - torch.lgamma(x_sub_tensor + 1)
            + r * log_r_over_r_plus_mu
            + x_sub_tensor * log_mu_over_r_plus_mu
        )
        .sum(dim=-1)
        .mean()
    )


def nb_loss_sparse(x_sparse, mu, log_r, row_idx=None):
    """Sparse-input equivalent of nb_loss: identical value, but the observed
    counts stay a sparse CSR tensor instead of being densified.

    The NB log-pmf's x=0 contribution, r*log(r/(r+mu)), is nonzero for every
    cell x gene, so it is summed densely over mu (which is dense regardless --
    the log-linear decoder predicts a rate for every gene). Only the x>0
    correction -- lgamma(x+r) - lgamma(r) - lgamma(x+1) + x*log(mu/(r+mu)) -- is
    gathered over the nonzeros of x. Summing the dense baseline and the sparse
    correction reproduces nb_loss's full sum over cells x genes; dividing by the
    batch size matches its .sum(dim=-1).mean() reduction.
    """
    r = torch.exp(log_r).clamp(min=1e-3)  # [n_genes]
    mu_nb = mu.float()
    eps = 1e-8
    r_plus_mu = r + mu_nb + eps  # [B, n_genes]

    # x=0 baseline over every cell x gene entry.
    baseline = (r * torch.log(r / r_plus_mu)).sum()

    col_idx = x_sparse.col_indices()
    if row_idx is None:
        crow = x_sparse.crow_indices()
        row_idx = torch.repeat_interleave(
            torch.arange(x_sparse.shape[0], device=mu.device), crow[1:] - crow[:-1]
        )
    x = x_sparse.values().float()
    r_nz = r[col_idx]
    mu_nz = mu_nb[row_idx, col_idx]
    correction = (
        torch.lgamma(x + r_nz)
        - torch.lgamma(r_nz)
        - torch.lgamma(x + 1)
        + x * torch.log((mu_nz + eps) / (r_nz + mu_nz + eps))
    ).sum()

    return -(baseline + correction) / mu.shape[0]


# Element budget for one chunk of the [nnz, conv_max+1] convolution tensor. Peak
# device memory for the correction is ~this many floats times a small constant
# (a handful of live intermediates + their gradients), independent of batch size
# and gene-panel width -- so a whole-transcriptome batch is processed in pieces
# rather than materializing one multi-GB tensor. ~8M elements keeps the peak well
# under 1 GB while amortizing Python/kernel-launch overhead over large chunks.
_CONV_CHUNK_ELEMENTS = 8_000_000


def _nbconv_logpx_chunk(x, r_nz, lam_nz, d_nz, conv_max, eps):
    """log P(Xhat = x) for a chunk of nonzeros, via the finite convolution
        P(Xhat = x) = sum_{j=0}^{min(x, conv_max)} NB(x - j; r, lam) * Poisson(j; delta),
    with j the contamination count and k = x - j the signal count. Shapes: all
    inputs [c]; returns [c]. Builds a [c, conv_max+1] grid."""
    j = torch.arange(conv_max + 1, device=x.device, dtype=x.dtype)  # [J+1]
    k = x.unsqueeze(1) - j.unsqueeze(0)  # [c, J+1]
    valid = k >= 0
    k_c = k.clamp(min=0.0)

    r_b = r_nz.unsqueeze(1)
    lam_b = lam_nz.unsqueeze(1)
    d_b = d_nz.unsqueeze(1)

    log_r_frac = torch.log(r_b / (r_b + lam_b + eps))
    log_lam_frac = torch.log(lam_b / (r_b + lam_b + eps))
    log_nb = (
        torch.lgamma(k_c + r_b)
        - torch.lgamma(r_b)
        - torch.lgamma(k_c + 1.0)
        + r_b * log_r_frac
        + k_c * log_lam_frac
    )
    log_pois = (
        -d_b
        + j.unsqueeze(0) * torch.log(d_b.clamp(min=eps))
        - torch.lgamma(j + 1.0).unsqueeze(0)
    )
    term = (log_nb + log_pois).masked_fill(~valid, float("-inf"))
    return torch.logsumexp(term, dim=1)  # [c]


def nbconv_loss_sparse(x_sparse, lam, delta, log_r, row_idx=None, conv_max=64):
    """Exact generative decontamination likelihood (paper's Option B / augmented
    model, marginalized in closed form).

    Models the observed count as a sum of two independent streams -- a negative-
    binomial signal and a Poisson contamination:

        u_cg   ~ Gamma(lam_cg, psi_g)              # latent true-signal rate
        A_cg   ~ Poisson(u_cg)          => A_cg ~ NB(mean=lam_cg, r)   (r = 1/psi_g)
        C_cg   ~ Poisson(delta_cg)                 # contamination (delta = alpha*inflow)
        Xhat_cg = A_cg + C_cg

    so the likelihood of the observed count is the convolution
        P(Xhat = x) = sum_{j=0}^{x} NB(x - j; r, lam) * Poisson(j; delta),
    which -- unlike NB(lam + delta, psi) -- keeps signal and contamination as
    distinct count processes. This is what lets the model reason about the latent
    split of each count (mechanisms 2-3 of next-step-generative-model.md): when
    contamination explains the counts, the signal posterior concentrates near 0
    and lam is genuinely unidentified, so its evidence for a nonzero beta is weak
    and the credible interval covers 0 rather than manufacturing a spurious call.
    The Poisson contamination carries no over-dispersion, so a count stream that
    looks Poisson is attributed to contamination, not signal.

    Structured exactly like nb_loss_sparse: a dense baseline over every cell x gene
    (the x=0 log-probability) plus a sparse correction gathered over the nonzeros of
    x. Because proseg guarantees inflow <= X elementwise, delta > 0 implies x > 0, so
    the entire correction lives on the nonzeros of x -- delta never contributes a
    dense correction of its own beyond the -delta term folded into the x=0 baseline.

    lam and delta are the *separate* signal and contamination rates (NOT their sum);
    lam must be strictly positive (exp of a clamped log-rate). delta may be None
    (no diffusion), in which case this reduces exactly to nb_loss_sparse.

    conv_max caps the convolution length, i.e. the number of counts attributable to
    contamination. It only needs to cover the Poisson(delta) tail (delta + a few sqrt
    (delta)) -- NOT the max observed count -- because the j <= x constraint is
    enforced by masking, so any entry with x <= conv_max is summed exactly regardless
    of conv_max. RegressionModel sizes it from the data's max inflow; the default 64
    is exact for inflow up to ~40 per entry. The [nnz, conv_max+1] correction is
    processed in nonzero chunks (see _CONV_CHUNK_ELEMENTS) so peak memory stays
    bounded on whole-transcriptome panels regardless of batch size.
    """
    r = torch.exp(log_r).clamp(min=1e-3)  # [n_genes]
    lam = lam.float()
    eps = 1e-8

    # x=0 baseline over every cell x gene entry:
    #   log P(X=0) = log NB(0; r, lam) + log Poisson(0; delta)
    #             = r*log(r/(r+lam))   +   (-delta)
    baseline = (r * torch.log(r / (r + lam + eps))).sum()
    if delta is not None:
        delta = delta.float()
        baseline = baseline - delta.sum()

    col_idx = x_sparse.col_indices()
    if row_idx is None:
        crow = x_sparse.crow_indices()
        row_idx = torch.repeat_interleave(
            torch.arange(x_sparse.shape[0], device=lam.device), crow[1:] - crow[:-1]
        )
    x = x_sparse.values().float()  # [nnz]
    r_nz = r[col_idx]  # [nnz]
    lam_nz = lam[row_idx, col_idx]  # [nnz]
    d_nz = (
        delta[row_idx, col_idx] if delta is not None else torch.zeros_like(x)
    )  # [nnz]

    # The correction replaces each nonzero's x=0 baseline term with its true
    # log-prob. Chunk over nonzeros so the [chunk, conv_max+1] grid never grows
    # with batch size or panel width.
    base_nz = r_nz * torch.log(r_nz / (r_nz + lam_nz + eps)) - d_nz
    nnz = x.shape[0]
    chunk = max(1, _CONV_CHUNK_ELEMENTS // (conv_max + 1))
    correction = lam.new_zeros(())
    for s in range(0, nnz, chunk):
        e = min(s + chunk, nnz)
        log_px = _nbconv_logpx_chunk(
            x[s:e], r_nz[s:e], lam_nz[s:e], d_nz[s:e], conv_max, eps
        )
        correction = correction + (log_px - base_nz[s:e]).sum()

    return -(baseline + correction) / lam.shape[0]


def kl_z(mu, logstd):
    """KL divergence from N(mu, exp(logstd)^2) to N(0, 1), averaged over cells."""
    return -0.5 * torch.sum(1 + 2 * logstd - mu.pow(2) - (2 * logstd).exp(), dim=-1).mean()


def size_factor_loss(log_sf, batch_log_sf_prior, sf_sigma):
    return (log_sf - batch_log_sf_prior).pow(2).mean() / (2 * sf_sigma**2)


def alpha_loss(log_alpha, alpha_reg):
    """L2 penalty on log(alpha), i.e. a log-normal prior on alpha = exp(log_alpha)
    pulling the shared inflow/outflow leak coefficient toward 1 -- the paper's
    "trust proseg's estimates at face value" reference point, with alpha free to
    move above or below 1 as the data demands. Used by the regression path;
    the factorization path uses the expression-weighted alpha_log_prior_loss."""
    return alpha_reg * log_alpha.pow(2).mean()


def alpha_kl(log_alpha_mu, log_alpha_logstd, m, prior_sigma=1.0):
    """KL from the variational posterior q(log_alpha)=N(mu, exp(logstd)^2) to the
    prior p(log_alpha)=N(0, prior_sigma^2) (alpha centered at 1). Summed over genes
    and divided by m (n_cells) to match kl_beta's per-cell scaling. Replaces the
    point-estimate alpha_loss when alpha is stochastic; propagates the
    contamination-vs-DE identifiability uncertainty into beta's posterior."""
    logstd = log_alpha_logstd.clamp(max=4.0)
    sigma2 = torch.exp(2.0 * logstd)
    tau2 = prior_sigma**2
    kl = 0.5 * (
        sigma2 / tau2
        + log_alpha_mu.pow(2) / tau2
        - 1.0
        + math.log(tau2)
        - 2.0 * logstd
    )
    return kl.sum() / m


def alpha_log_prior_loss(log_alpha, alpha_reg, gene_expression=None):
    """L2 penalty on log(alpha), i.e. a log-normal prior on alpha = exp(log_alpha)
    centered at 1 -- the paper's "trust Proseg's estimates at face value" reference
    point, with alpha free to move above 1 (Proseg underestimated leakage) or below
    (Proseg overestimated it) as the data demands.

    If gene_expression (mean count per cell) is given, the penalty is weighted by
    1/sqrt(expression+1), same as the older alpha_loss: low-count genes keep close
    to the full baseline pull toward alpha=1, while high-count genes -- where there's
    enough signal to actually distinguish contamination from noise -- get a weaker
    pull and more freedom to deviate. Without this, a single global alpha_reg applies
    equally regardless of how much evidence a gene's counts can support, which let
    low-count genes get corrected just as aggressively as well-supported ones and
    measurably degraded separation of rarer cell types relying on them.
    """
    if gene_expression is not None:
        weight = 1.0 / torch.sqrt(gene_expression + 1.0)
        return alpha_reg * (log_alpha ** 2 * weight).sum()
    return alpha_reg * (log_alpha ** 2).sum()


def metagene_entropy_loss(u, sparsity_reg, eps=1e-8):
    """Entropy penalty on each cell's normalized metagene loadings (u / sum(u)),
    encouraging concentration onto few metagenes rather than spreading contamination
    across several factors. sparsity_reg should be annealed in over training."""
    pi = u / (u.sum(dim=1, keepdim=True) + eps)
    entropy = -(pi * torch.log(pi.clamp(min=eps))).sum(dim=1)
    return sparsity_reg * entropy.mean()


def dirichlet_purity_loss(
    composition: torch.Tensor, weight: float, alpha: float = 0.5, eps: float = 1e-8
) -> torch.Tensor:
    """Negative log-density (dropping the pi-independent normalizing constant)
    of a symmetric Dirichlet(alpha) prior over each cell's metagene composition
    (a simplex vector, e.g. SparseCompositionAbundanceEncoder's composition
    output -- NOT u itself, which also carries the abundance scale and so
    isn't on the simplex).

    log p(pi | alpha) = const + (alpha - 1) * sum_k log(pi_k)

    For alpha < 1 this density is bathtub-shaped: it blows up toward the
    simplex corners (one pi_k -> 1, the rest -> 0) and is lowest at the
    uniform center, so minimizing the returned loss (the negative log
    density) continuously pulls composition toward purity.

    Unlike sparsemax/entmax's hard sparsity (see DIFFUSION_INVESTIGATION_NOTES.md
    sec. 18-20), this never creates a literal zero-gradient trap: composition
    is expected to come from softmax (always strictly positive), and this
    penalty just adds smooth, everywhere-nonzero pressure toward (but never
    exactly reaching) a corner -- so nothing can get permanently excluded the
    way v's entmax/sparsemax entries could.

    weight controls the prior's strength relative to the reconstruction loss;
    with alpha < 1 the loss is intentionally unbounded below as composition
    sharpens, so weight needs to be tuned rather than left arbitrarily large.
    """
    return weight * (1 - alpha) * torch.log(composition.clamp(min=eps)).sum(dim=1).mean()


def gamma_prior_loss(log_r, alpha, beta, r_weight):
    """Negative log Gamma(alpha, beta) prior on dispersion r = exp(log_r).

    Weighted by r_weight = 1/n_batches so the dataset-level prior is counted
    exactly once per epoch under mini-batch SGD.
    """
    r = torch.exp(log_r)
    log_prior = (
        alpha * math.log(beta)
        - math.lgamma(alpha)
        + (alpha - 1) * torch.log(r)
        - beta * r
    )
    return -log_prior.sum() * r_weight


def metagene_correlation_loss(H_constrained, strength):
    """Pearson correlation penalty between rows of the gene program matrix H.

    Encourages diverse, non-redundant factors by penalizing squared off-diagonal
    elements of the correlation matrix.
    """
    k = H_constrained.shape[0]
    if k <= 1:
        return torch.zeros((), device=H_constrained.device)
    h_centered = H_constrained - H_constrained.mean(dim=1, keepdim=True)
    row_norms = torch.sqrt((h_centered**2).sum(dim=1, keepdim=True))
    h_normalized = h_centered / (row_norms + 1e-8)
    corr_matrix = h_normalized @ h_normalized.T
    mask = 1.0 - torch.eye(k, device=H_constrained.device)
    return strength * ((corr_matrix * mask) ** 2).sum()


def gene_floor_loss(v_raw: torch.Tensor, weight: float, margin: float = -3.0) -> torch.Tensor:
    """Anti-collapse regularizer for FactorizationVAE.v (the raw, pre-activation
    metagene-program parameter), for use with the sparsemax/entmax15
    metagene_activation options.

    Both sparsemax and entmax give exactly zero gradient to v[k, g] whenever
    gene g falls outside factor k's support -- so once every factor's
    independent per-row optimization happens to exclude a low-signal gene
    (observed for B-cell markers under entmax15: excluded from all 50 factors,
    not just the wrong ones -- DIFFUSION_INVESTIGATION_NOTES.md sec. 20), there
    is no gradient path back through the normal reconstruction loss to recover
    it. This penalizes each gene's best-positioned factor directly on the raw
    parameter (bypassing the clamp entirely, so it always has a live gradient,
    even for genes currently excluded everywhere): for each gene, shift every
    factor's row by its own max (matching entmax/sparsemax's internal
    shift-invariant normalization) and penalize the best (least-negative)
    shifted score if it falls below `margin` -- i.e. "every gene should be
    within `margin` of at least one factor's own top choice," not "every gene
    should have some absolute raw score." Zero cost once a gene clears the
    margin somewhere, so it does not fight genuine sparsification.
    """
    row_shifted = v_raw - v_raw.max(dim=1, keepdim=True).values
    gene_best = row_shifted.max(dim=0).values
    return weight * F.relu(margin - gene_best).pow(2).sum()


def min_volume_loss(H: torch.Tensor, strength: float, delta: float = 1e-3) -> torch.Tensor:
    """Minimum-volume regularization on metagene programs H (k x n, e.g. v_norm()).

    Penalizes log-det of the Gram matrix H @ H^T + delta*I. sqrt(det(H H^T)) is
    proportional to the k-dimensional volume of the simplex spanned by H's rows, so
    minimizing this pulls the metagenes toward the smallest simplex that can still
    explain the data -- discouraging redundant/overlapping factors that let
    ambiguous cells blend between them for free, without penalizing genuine
    diversity the reconstruction loss actually needs.
    """
    k = H.shape[0]
    gram = H @ H.T + delta * torch.eye(k, device=H.device, dtype=H.dtype)
    _, logdet = torch.linalg.slogdet(gram)
    return strength * logdet
