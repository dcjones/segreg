import math

import torch
import torch.nn.functional as F


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


def kl_z(mu, logstd):
    """KL divergence from N(mu, exp(logstd)^2) to N(0, 1), averaged over cells."""
    return -0.5 * torch.sum(1 + 2 * logstd - mu.pow(2) - (2 * logstd).exp(), dim=-1).mean()


def size_factor_loss(log_sf, batch_log_sf_prior, sf_sigma):
    return (log_sf - batch_log_sf_prior).pow(2).mean() / (2 * sf_sigma**2)


def alpha_loss(log_alpha, alpha_reg):
    """Penalty pulling alpha_g (shared inflow/outflow leak coefficient) toward 1."""
    return alpha_reg * log_alpha.pow(2).mean()


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
