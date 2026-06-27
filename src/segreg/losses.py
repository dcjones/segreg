import torch
import torch.nn.functional as F


def nb_loss(x_sub_tensor, mu, log_r):
    """Negative binomial reconstruction loss, summed over genes, averaged over cells.

    Parameterized by r (= 1 / psi_g in the paper's dispersion notation), so that
    Var[X] = mu + mu^2 / r = mu + psi_g * mu^2.
    """
    r = F.softplus(log_r).clamp(min=1e-3)
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
