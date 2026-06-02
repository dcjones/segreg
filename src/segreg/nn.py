import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Node-based Encoder with LayerNorm.
    Takes gene counts and design covariates,
    and outputs parameters for the latent normal distribution (mu, logstd).
    """

    def __init__(
        self, in_channels: int, hidden_channels: int = 128, latent_dim: int = 64
    ):
        super().__init__()
        self.lin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
        )

        self.mu_head = nn.Linear(hidden_channels, latent_dim)
        self.logstd_head = nn.Linear(hidden_channels, latent_dim)

    def forward(self, x):
        h = self.lin(x)
        mu = self.mu_head(h)
        logstd = self.logstd_head(h)
        return mu, logstd


class NodeDecoder(nn.Module):
    """
    Decodes the node latent representation back into unconstrained expression rates (rho).
    Uses a simple linear transformation to capture residual biological variation
    as linear gene modules.
    """

    def __init__(self, latent_dim: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.lin = nn.Linear(latent_dim, out_channels, bias=False)

    def forward(self, z):
        return self.lin(z)


class SegregBase(nn.Module):
    """Shared VAE backbone: encoder, decoder, gene_bias, log_r, size factor, inflow scale."""

    def __init__(
        self,
        n_cells: int,
        n_genes: int,
        encoder_in_channels: int,
        hidden_channels: int = 64,
        latent_dim: int = 32,
        rate_offset: float = 1e-2,
        include_diffusion: bool = True,
        include_size_factor: bool = True,
        log_mean_expr: torch.Tensor | None = None,
        log_sf_prior: torch.Tensor | None = None,
    ):
        super().__init__()
        self.include_diffusion = include_diffusion
        self.include_size_factor = include_size_factor
        self.rate_offset = rate_offset

        self.encoder = Encoder(encoder_in_channels, hidden_channels, latent_dim)
        self.node_decoder = NodeDecoder(latent_dim, hidden_channels, n_genes)

        if self.include_size_factor:
            self.log_sf_embed = nn.Embedding(n_cells, 1)
            if log_sf_prior is not None:
                self.log_sf_embed.weight.data = log_sf_prior.clone().unsqueeze(-1)

        if log_mean_expr is not None:
            self.gene_bias = nn.Parameter(log_mean_expr.clone())
        else:
            self.gene_bias = nn.Parameter(torch.zeros(n_genes))

        self.log_r = nn.Parameter(torch.full((n_genes,), 9.3))

        if include_diffusion:
            # Per-gene learnable inflow scale: proseg inflow may systematically underestimate
            # contamination for highly expressed neighboring-cell genes. A learned scale
            # is identifiable because contamination genes are under-fitted (x_hat < x),
            # while genuine DE genes are already well-explained by beta/z and stay near 1.
            self.log_inflow_scale = nn.Parameter(torch.zeros(n_genes))

    def reparameterize(self, mu, logstd):
        if self.training:
            std = torch.exp(logstd.clamp(max=8.0))
            eps = torch.randn_like(std)
            return (mu + eps * std).clamp(-40.0, 40.0)
        return mu.clamp(-40.0, 40.0)

    def prepare_encoder_input(
        self, x_sub_tensor: torch.Tensor, batch_idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Normalize counts and compute size factor. Returns (encoder_in, log_sf)."""
        if self.include_size_factor:
            log_sf = self.log_sf_embed(batch_idx).squeeze(-1)
            x_norm = x_sub_tensor / (torch.exp(log_sf).unsqueeze(-1) + 1e-8) * 1000.0
            encoder_in = torch.log1p(x_norm)
        else:
            log_sf = None
            encoder_in = torch.log1p(x_sub_tensor)
        return encoder_in, log_sf


class SegregVAE(SegregBase):
    def __init__(
        self,
        n_cells: int,
        n_genes: int,
        n_covariates: int,
        hidden_channels: int = 64,
        latent_dim: int = 32,
        kappa: float = 100.0,
        rate_offset: float = 1e-2,
        include_diffusion: bool = True,
        include_size_factor: bool = True,
        log_mean_expr: torch.Tensor | None = None,
        log_sf_prior: torch.Tensor | None = None,
        beta_init: torch.Tensor | None = None,
        beta_logstd_init: torch.Tensor | None = None,
    ):
        encoder_in_channels = n_genes if include_size_factor else n_genes + n_covariates
        super().__init__(
            n_cells=n_cells,
            n_genes=n_genes,
            encoder_in_channels=encoder_in_channels,
            hidden_channels=hidden_channels,
            latent_dim=latent_dim,
            rate_offset=rate_offset,
            include_diffusion=include_diffusion,
            include_size_factor=include_size_factor,
            log_mean_expr=log_mean_expr,
            log_sf_prior=log_sf_prior,
        )

        if beta_init is not None:
            self.beta_mu = nn.Parameter(beta_init.clone())
        else:
            self.beta_mu = nn.Parameter(torch.zeros(n_covariates, n_genes))
        if beta_logstd_init is not None:
            self.beta_logstd = nn.Parameter(beta_logstd_init.clone())
        else:
            self.beta_logstd = nn.Parameter(torch.full((n_covariates, n_genes), -3.0))

    def forward(self, x, covariates, inflow, outflow, log_size_factor=None):
        mu, logstd = self.encoder(x)
        z = self.reparameterize(mu, logstd)
        rho = self.node_decoder(z)

        if self.training:
            beta_std = torch.exp(self.beta_logstd.clamp(max=4.0))
            beta = self.beta_mu + torch.randn_like(beta_std) * beta_std
        else:
            beta = self.beta_mu

        log_rate = rho + covariates @ beta + self.gene_bias
        if self.include_size_factor and log_size_factor is not None:
            log_rate = log_rate + log_size_factor.unsqueeze(-1)
        lam = torch.exp(torch.clamp(log_rate, min=-15.0, max=15.0))

        if self.include_diffusion:
            inflow_scale = torch.exp(self.log_inflow_scale)
            x_hat = lam + inflow_scale * inflow + self.rate_offset
        else:
            x_hat = lam + self.rate_offset

        return x_hat, mu, logstd, beta


class SegregFactorizationVAE(SegregBase):
    def __init__(
        self,
        n_cells: int,
        n_genes: int,
        n_factors: int,
        hidden_channels: int = 64,
        latent_dim: int = 32,
        rate_offset: float = 1e-2,
        include_diffusion: bool = True,
        include_size_factor: bool = True,
        log_mean_expr: torch.Tensor | None = None,
        log_sf_prior: torch.Tensor | None = None,
    ):
        # W is per-cell and looked up by index, not fed to the encoder
        super().__init__(
            n_cells=n_cells,
            n_genes=n_genes,
            encoder_in_channels=n_genes,
            hidden_channels=hidden_channels,
            latent_dim=latent_dim,
            rate_offset=rate_offset,
            include_diffusion=include_diffusion,
            include_size_factor=include_size_factor,
            log_mean_expr=log_mean_expr,
            log_sf_prior=log_sf_prior,
        )
        self.n_factors = n_factors
        self.W_embed = nn.Embedding(n_cells, n_factors)
        nn.init.normal_(self.W_embed.weight, std=0.1)
        self.H = nn.Parameter(torch.randn(n_factors, n_genes) * 0.01)

    def forward(
        self,
        encoder_in: torch.Tensor,
        batch_idx: torch.Tensor,
        inflow: torch.Tensor | None,
        outflow: torch.Tensor | None,
        log_size_factor: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logstd = self.encoder(encoder_in)
        z = self.reparameterize(mu, logstd)
        rho = self.node_decoder(z)

        W = self.W_embed(batch_idx)  # (batch, n_factors)
        log_rate = rho + W @ self.H + self.gene_bias
        if self.include_size_factor and log_size_factor is not None:
            log_rate = log_rate + log_size_factor.unsqueeze(-1)
        lam = torch.exp(torch.clamp(log_rate, min=-15.0, max=15.0))

        if self.include_diffusion:
            inflow_scale = torch.exp(self.log_inflow_scale)
            x_hat = lam + inflow_scale * inflow + self.rate_offset
        else:
            x_hat = lam + self.rate_offset

        return x_hat, mu, logstd, W
