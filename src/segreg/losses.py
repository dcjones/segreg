import torch
import torch.nn.functional as F


def nb_loss(x_sub_tensor, x_hat, log_r):
    """Negative binomial reconstruction loss, summed over genes, averaged over cells."""
    r = F.softplus(log_r).clamp(min=1e-3)
    mu_nb = x_hat.float()
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


def inflow_scale_loss(log_inflow_scale, inflow_scale_reg):
    return inflow_scale_reg * log_inflow_scale.pow(2).mean()
