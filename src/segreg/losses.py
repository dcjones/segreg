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
