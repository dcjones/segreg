import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Stochastic encoder for SegregVAE.
    Takes normalized gene counts and outputs parameters for the latent normal
    distribution (mu, logstd).

    Only the first linear layer touches the [B, n_genes] input, so it is the one
    place worth keeping sparse: when given a sparse CSR input it runs that layer
    as torch.sparse.mm (against the same nn.Linear weight, so numerics and
    parameters are identical to the dense path), avoiding densification of the
    92%-zero count batch. Every downstream layer operates on the dense
    [B, hidden] activations as before.
    """

    def __init__(
        self, in_channels: int, hidden_channels: int = 128, latent_dim: int = 64
    ):
        super().__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.rest = nn.Sequential(
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
        )

        self.mu_head = nn.Linear(hidden_channels, latent_dim)
        self.logstd_head = nn.Linear(hidden_channels, latent_dim)

    def forward(self, x):
        if x.layout == torch.sparse_csr:
            # torch.sparse.mm(x, W) with W = weight.t() reproduces nn.Linear's
            # x @ weight.t() + bias exactly, but consumes the sparse batch directly.
            # Forced to fp32: sparse CSR matmul (and its sampled_addmm backward) does
            # not support the mixed bf16/fp32 operands autocast would introduce.
            with torch.autocast(device_type=x.device.type, enabled=False):
                w = self.lin1.weight.t().float()
                h = torch.sparse.mm(x.float(), w) + self.lin1.bias.float()
        else:
            h = self.lin1(x)
        h = self.rest(h)
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


class SegregVAE(nn.Module):
    """Segmentation-aware log-linear regression VAE.

    A per-cell amortized latent z (via the stochastic Encoder / NodeDecoder)
    plus a log-linear regression on the design matrix produce the own-signal
    rate lambda; outflow enters multiplicatively as a retention factor
    exp(-alpha_g * phi_cg) and inflow additively as alpha_g * inflow_cg, per the
    paper's segmentation-aware count model.
    """

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
        super().__init__()
        self.include_diffusion = include_diffusion
        self.include_size_factor = include_size_factor
        self.rate_offset = rate_offset

        encoder_in_channels = n_genes if include_size_factor else n_genes + n_covariates
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
            # alpha_g: shared per-gene leak coefficient calibrating how far to trust
            # proseg's inflow/outflow estimates. alpha_g = 1 takes them at face value;
            # it's identifiable because contamination genes are under-fitted by
            # beta/z alone, while genuine DE genes already fit well and keep alpha
            # near 1. alpha = exp(log_alpha) >= 0 is unbounded above so alpha_g > 1
            # can absorb leakage proseg underestimates (a sigmoid bound to (0,1)
            # could not). Prior centered at alpha_g = 1, i.e. log_alpha = 0.
            self.log_alpha = nn.Parameter(torch.zeros(n_genes))

        if beta_init is not None:
            self.beta_mu = nn.Parameter(beta_init.clone())
        else:
            self.beta_mu = nn.Parameter(torch.zeros(n_covariates, n_genes))
        if beta_logstd_init is not None:
            self.beta_logstd = nn.Parameter(beta_logstd_init.clone())
        else:
            self.beta_logstd = nn.Parameter(torch.full((n_covariates, n_genes), -3.0))

    def reparameterize(self, mu, logstd):
        if self.training:
            std = torch.exp(logstd.clamp(max=8.0))
            eps = torch.randn_like(std)
            return (mu + eps * std).clamp(-40.0, 40.0)
        return mu.clamp(-40.0, 40.0)

    def prepare_encoder_input(
        self, x_sub_tensor: torch.Tensor, batch_idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Normalize counts and compute size factor. Returns (encoder_in, log_sf).

        Accepts either a dense [B, n_genes] tensor or a sparse CSR batch. For
        sparse input the per-cell size-factor rescale and log1p are applied to the
        nonzero values only (log1p(0) = 0, so zeros are unaffected), keeping the
        result sparse for the encoder's first layer.
        """
        log_sf = (
            self.log_sf_embed(batch_idx).squeeze(-1)
            if self.include_size_factor
            else None
        )

        if x_sub_tensor.layout == torch.sparse_csr:
            values = x_sub_tensor.values()
            if log_sf is not None:
                crow = x_sub_tensor.crow_indices()
                row = torch.repeat_interleave(
                    torch.arange(x_sub_tensor.shape[0], device=x_sub_tensor.device),
                    crow[1:] - crow[:-1],
                )
                scale = 1000.0 / (torch.exp(log_sf) + 1e-8)  # [B]
                values = values * scale[row]
            encoder_in = torch.sparse_csr_tensor(
                x_sub_tensor.crow_indices(),
                x_sub_tensor.col_indices(),
                torch.log1p(values),
                size=x_sub_tensor.shape,
                device=x_sub_tensor.device,
            )
        else:
            if log_sf is not None:
                x_norm = x_sub_tensor / (torch.exp(log_sf).unsqueeze(-1) + 1e-8) * 1000.0
                encoder_in = torch.log1p(x_norm)
            else:
                encoder_in = torch.log1p(x_sub_tensor)
        return encoder_in, log_sf

    def forward(self, x, covariates, inflow, phi, log_size_factor=None):
        z_mu, z_logstd = self.encoder(x)
        z = self.reparameterize(z_mu, z_logstd)
        z_offset = self.node_decoder(z)

        if self.training:
            beta_std = torch.exp(self.beta_logstd.clamp(max=4.0))
            beta = self.beta_mu + torch.randn_like(beta_std) * beta_std
        else:
            beta = self.beta_mu

        log_rate = z_offset + covariates @ beta + self.gene_bias
        if self.include_size_factor and log_size_factor is not None:
            log_rate = log_rate + log_size_factor.unsqueeze(-1)
        lam = torch.exp(torch.clamp(log_rate, min=-15.0, max=15.0))

        if self.include_diffusion:
            assert inflow is not None and phi is not None
            # retention = exp(-alpha*phi) is 1 wherever phi=0, so mu is dense
            # regardless; densify the (sparse) inflow/phi batches here to build it.
            if phi.layout == torch.sparse_csr:
                phi = phi.to_dense()
            if inflow.layout == torch.sparse_csr:
                inflow = inflow.to_dense()
            alpha = torch.exp(self.log_alpha)
            retention = torch.exp(-alpha * phi)
            delta = alpha * inflow
            mu = retention * lam + delta + self.rate_offset
        else:
            mu = lam + self.rate_offset

        return mu, z_mu, z_logstd, beta
