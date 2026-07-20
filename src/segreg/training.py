import math

import torch
import torch.nn as nn

from .losses import (
    alpha_kl,
    alpha_loss,
    kl_z,
    nb_loss_sparse,
    size_factor_loss,
)
from .nn import SegregVAE


class SegregTrainingWrapper(nn.Module):
    def __init__(
        self,
        model: SegregVAE,
        beta_prior_scale: torch.Tensor,
        m: int,
        sf_sigma: float,
        alpha_reg: float = 0.1,
    ):
        super().__init__()
        self.model = model
        self.register_buffer("beta_prior_scale", beta_prior_scale)
        self.m = m
        self.sf_sigma = sf_sigma
        self.alpha_reg = alpha_reg

    def forward(
        self,
        batch_idx,
        batch_x,
        batch_log_sf_prior,
        x_sparse,
        inflow_sub,
        phi_sub,
        row_idx,
        current_beta_kl: torch.Tensor,
        batch_component=None,
    ):
        encoder_in, log_sf = self.model.prepare_encoder_input(x_sparse, batch_idx)

        mu_hat, z_mu, z_logstd, beta = self.model(
            encoder_in,
            batch_x,
            inflow=inflow_sub,
            phi=phi_sub,
            log_size_factor=log_sf,
            component=batch_component,
        )

        loss_recon = nb_loss_sparse(x_sparse, mu_hat, self.model.log_r, row_idx=row_idx)
        kl_z_val = kl_z(z_mu, z_logstd)

        beta_f = beta.float()
        gamma = self.beta_prior_scale
        log_q = (
            -0.5
            * ((beta_f - self.model.beta_mu) / torch.exp(self.model.beta_logstd)).pow(2)
            - self.model.beta_logstd
            - 0.5 * math.log(2.0 * math.pi)
        )
        log_p = (
            -math.log(math.pi) - torch.log(gamma) - torch.log1p((beta_f / gamma).pow(2))
        )
        kl_beta = (log_q - log_p).sum() / self.m

        if self.model.include_size_factor and log_sf is not None:
            loss_sf = size_factor_loss(log_sf, batch_log_sf_prior, self.sf_sigma)
        else:
            loss_sf = torch.tensor(0.0, device=mu_hat.device)

        if self.model.include_diffusion:
            if getattr(self.model, "log_alpha_logstd", None) is not None:
                # Variational alpha: KL to N(0, 1) prior (alpha centered at 1),
                # scaled by alpha_reg. Propagates alpha uncertainty into beta.
                loss_alpha = self.alpha_reg * alpha_kl(
                    self.model.log_alpha, self.model.log_alpha_logstd, self.m
                )
            else:
                loss_alpha = alpha_loss(self.model.log_alpha, self.alpha_reg)
            if getattr(self.model, "log_alpha_ret", None) is not None:
                loss_alpha = loss_alpha + alpha_loss(
                    self.model.log_alpha_ret, self.alpha_reg
                )
        else:
            loss_alpha = torch.tensor(0.0, device=mu_hat.device)

        loss = (
            loss_recon + (current_beta_kl * kl_z_val) + kl_beta + loss_sf + loss_alpha
        )
        return loss, loss_recon
