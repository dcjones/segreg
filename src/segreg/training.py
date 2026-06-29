import math

import torch
import torch.nn as nn

from .losses import (
    alpha_loss,
    gamma_prior_loss,
    kl_z,
    metagene_correlation_loss,
    nb_loss,
    poisson_loss,
    size_factor_loss,
)
from .nn import SegregFactorizationVAE, SegregVAE


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
        x_sub_tensor,
        inflow_sub,
        phi_sub,
        current_beta_kl: torch.Tensor,
    ):
        encoder_in, log_sf = self.model.prepare_encoder_input(x_sub_tensor, batch_idx)

        mu_hat, z_mu, z_logstd, beta = self.model(
            encoder_in,
            batch_x,
            inflow=inflow_sub,
            phi=phi_sub,
            log_size_factor=log_sf,
        )

        loss_recon = nb_loss(x_sub_tensor, mu_hat, self.model.log_r)
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
            loss_alpha = alpha_loss(self.model.log_alpha, self.alpha_reg)
        else:
            loss_alpha = torch.tensor(0.0, device=mu_hat.device)

        loss = loss_recon + (current_beta_kl * kl_z_val) + kl_beta + loss_sf + loss_alpha
        return loss, loss_recon


class FactorizationTrainingWrapper(nn.Module):
    def __init__(
        self,
        model: SegregFactorizationVAE,
        m: int,
        sf_sigma: float,
        r_weight: float,
        r_prior_alpha: float = 2.0,
        r_prior_beta: float = 2.0,
        alpha_reg: float = 1.0,
        metagene_reg_strength: float = 0.01,
        likelihood: str = "poisson",
        gene_expression: torch.Tensor | None = None,
    ):
        super().__init__()
        self.model = model
        self.m = m
        self.sf_sigma = sf_sigma
        self.r_weight = r_weight
        self.r_prior_alpha = r_prior_alpha
        self.r_prior_beta = r_prior_beta
        self.alpha_reg = alpha_reg
        self.metagene_reg_strength = metagene_reg_strength
        self.likelihood = likelihood
        if gene_expression is not None:
            self.register_buffer("gene_expression", gene_expression)
        else:
            self.gene_expression = None

    def forward(
        self,
        batch_idx,
        batch_log_sf_prior,
        x_sub_tensor,
        inflow_sub,
        phi_sub,
    ):
        _, log_sf = self.model.prepare_encoder_input(x_sub_tensor, batch_idx)

        mu_hat, W, H = self.model(
            batch_idx,
            inflow=inflow_sub,
            phi=phi_sub,
            log_size_factor=log_sf,
        )

        if self.likelihood == "poisson":
            loss_recon = poisson_loss(x_sub_tensor, mu_hat)
            loss_r_prior = torch.tensor(0.0, device=mu_hat.device)
        else:
            loss_recon = nb_loss(x_sub_tensor, mu_hat, self.model.log_r)
            loss_r_prior = gamma_prior_loss(
                self.model.log_r, self.r_prior_alpha, self.r_prior_beta, self.r_weight
            )

        if self.model.include_size_factor and log_sf is not None:
            loss_sf = size_factor_loss(log_sf, batch_log_sf_prior, self.sf_sigma)
        else:
            loss_sf = torch.tensor(0.0, device=mu_hat.device)

        if self.model.include_diffusion:
            gene_expr = self.gene_expression.to(mu_hat.device) if self.gene_expression is not None else None
            loss_alpha = alpha_loss(self.model.log_alpha, self.alpha_reg, gene_expr)
        else:
            loss_alpha = torch.tensor(0.0, device=mu_hat.device)

        loss_metagene = metagene_correlation_loss(H, self.metagene_reg_strength)

        loss = loss_recon + loss_r_prior + loss_sf + loss_alpha + loss_metagene
        return loss, loss_recon
