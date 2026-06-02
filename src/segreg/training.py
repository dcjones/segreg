import math

import torch
import torch.nn as nn

from .losses import inflow_scale_loss, kl_z, nb_loss, size_factor_loss
from .nn import SegregFactorizationVAE, SegregVAE


class SegregTrainingWrapper(nn.Module):
    def __init__(
        self,
        model: SegregVAE,
        beta_prior_scale: torch.Tensor,
        m: int,
        sf_sigma: float,
        inflow_scale_reg: float = 0.1,
    ):
        super().__init__()
        self.model = model
        self.register_buffer("beta_prior_scale", beta_prior_scale)
        self.m = m
        self.sf_sigma = sf_sigma
        self.inflow_scale_reg = inflow_scale_reg

    def forward(
        self,
        batch_idx,
        batch_x,
        batch_log_sf_prior,
        x_sub_tensor,
        inflow_sub,
        outflow_sub,
        current_beta_kl: torch.Tensor,
    ):
        encoder_in, log_sf = self.model.prepare_encoder_input(x_sub_tensor, batch_idx)

        x_hat, mu, logstd, beta = self.model(
            encoder_in,
            batch_x,
            inflow=inflow_sub,
            outflow=outflow_sub,
            log_size_factor=log_sf,
        )

        loss_recon = nb_loss(x_sub_tensor, x_hat, self.model.log_r)
        kl_z_val = kl_z(mu, logstd)

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
            loss_sf = torch.tensor(0.0, device=x_hat.device)

        if self.model.include_diffusion:
            loss_inflow_scale = inflow_scale_loss(
                self.model.log_inflow_scale, self.inflow_scale_reg
            )
        else:
            loss_inflow_scale = torch.tensor(0.0, device=x_hat.device)

        loss = loss_recon + (current_beta_kl * kl_z_val) + kl_beta + loss_sf + loss_inflow_scale
        return loss, loss_recon


class FactorizationTrainingWrapper(nn.Module):
    def __init__(
        self,
        model: SegregFactorizationVAE,
        m: int,
        sf_sigma: float,
        w_reg: float = 1.0,
        h_reg: float = 1.0,
        inflow_scale_reg: float = 0.1,
    ):
        super().__init__()
        self.model = model
        self.m = m
        self.sf_sigma = sf_sigma
        self.w_reg = w_reg
        self.h_reg = h_reg
        self.inflow_scale_reg = inflow_scale_reg

    def forward(
        self,
        batch_idx,
        batch_log_sf_prior,
        x_sub_tensor,
        inflow_sub,
        outflow_sub,
        current_kl_weight: torch.Tensor,
    ):
        encoder_in, log_sf = self.model.prepare_encoder_input(x_sub_tensor, batch_idx)

        x_hat, mu, logstd, W = self.model(
            encoder_in,
            batch_idx,
            inflow=inflow_sub,
            outflow=outflow_sub,
            log_size_factor=log_sf,
        )

        loss_recon = nb_loss(x_sub_tensor, x_hat, self.model.log_r)
        kl_z_val = kl_z(mu, logstd)

        loss_w = self.w_reg * W.pow(2).mean()
        loss_h = self.h_reg * self.model.H.pow(2).mean()

        if self.model.include_size_factor and log_sf is not None:
            loss_sf = size_factor_loss(log_sf, batch_log_sf_prior, self.sf_sigma)
        else:
            loss_sf = torch.tensor(0.0, device=x_hat.device)

        if self.model.include_diffusion:
            loss_inflow_scale = inflow_scale_loss(
                self.model.log_inflow_scale, self.inflow_scale_reg
            )
        else:
            loss_inflow_scale = torch.tensor(0.0, device=x_hat.device)

        loss = loss_recon + current_kl_weight * kl_z_val + loss_w + loss_h + loss_sf + loss_inflow_scale
        return loss, loss_recon
