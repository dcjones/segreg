import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Stochastic encoder for SegregVAE.
    Takes normalized gene counts (dense [B, n_genes]) and outputs parameters for
    the latent normal distribution (mu, logstd).

    The first layer's [B, n_genes] input is densified upstream rather than fed as a
    sparse CSR tensor: benchmarking across 500-18,006 genes showed a bf16
    tensor-core GEMM beats torch.sparse.mm at every panel size in this range (the
    ~6%-dense CSR SpMM never recovers the tensor-core throughput it gives up). The
    sparse batch is still transferred compactly and reused by the sparse NB
    likelihood, where sparsity does pay off; only the encoder matmul is dense.
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
        include_retention: bool = True,
        include_delta: bool = True,
        separate_retention_alpha: bool = False,
        stochastic_alpha: bool = False,
        component_alpha: bool = False,
        n_components: int = 1,
        likelihood: str = "nb_mean",
        contam_var: str = "poisson",
        retention_form: str | None = None,
        encoder_input: str = "raw",
        conv_max: int = 64,
        include_size_factor: bool = True,
        log_mean_expr: torch.Tensor | None = None,
        log_sf_prior: torch.Tensor | None = None,
        beta_init: torch.Tensor | None = None,
        beta_logstd_init: torch.Tensor | None = None,
    ):
        super().__init__()
        # likelihood: "nb_mean" is the phenomenological NB placed on the combined
        # mean mu = retention*lam + delta (the paper's default, closed only in the
        # mean). "nb_conv" is the exact generative marginal of the augmented model
        # (next-step-generative-model.md, Option B): Xhat = A + C with A ~ NB(lam,
        # psi) signal and C ~ Poisson(delta) contamination, so the likelihood is the
        # convolution NB(lam,psi) (+) Poisson(delta) rather than an NB on their sum.
        # "nb_mm" is the moment-matched approximation to nb_conv: one NB whose mean
        # and variance match the two-stream model's (see nbmm_loss_sparse). It costs
        # an NB evaluation instead of a convolution, but cannot represent
        # under-dispersion, so it only partially captures a sub-Poisson contamination
        # variance.
        #
        # In nb_conv/nb_mm mode alpha is FIXED at 1 (delta = inflow taken at face
        # value): with delta a known contamination rate there is no alpha-vs-beta
        # competition, the structural degeneracy that biases beta negative for
        # contamination-dominated should-be-null genes. A free alpha would also make
        # the variance bookkeeping ill-defined (scaling a count rate by alpha scales
        # its variance by alpha^2, which proseg's estimate does not track). The alpha
        # variational/component/separate machinery is therefore disabled in these
        # modes. Retention is NOT: in nb_conv/nb_mm it multiplies the signal rate
        # inside the two-stream model, mean(A) = r_cg * lam_cg, exactly as the paper
        # writes it. (Retention was previously benchmarked as inert, but only under
        # nb_mean, where it competed with a free alpha_g for the same outflow signal.)
        if likelihood not in ("nb_mean", "nb_conv", "nb_mm"):
            raise ValueError(f"unknown likelihood {likelihood!r}")
        # contam_var selects what variance the contamination count C is given, i.e.
        # how proseg's expected_inflow_var (V_f) is read:
        #   "poisson"     Var[C] = delta            -- ignore V_f (original nb_conv).
        #   "proseg"      Var[C] = V_f              -- V_f is the posterior variance
        #                 of the contaminant COUNT (a Poisson-binomial over the
        #                 cell's own transcripts), which is sub-Poisson; represented
        #                 by a binomial contamination arm.
        #   "proseg_rate" Var[C] = delta + V_f      -- V_f is epistemic uncertainty
        #                 about the contamination RATE (f ~ Gamma), stacked on
        #                 Poisson sampling noise; an NB contamination arm.
        if contam_var not in ("poisson", "proseg", "proseg_rate"):
            raise ValueError(f"unknown contam_var {contam_var!r}")
        self.likelihood = likelihood
        self.contam_var = contam_var
        self.contam_family = {
            "poisson": "poisson",
            "proseg": "binomial",
            "proseg_rate": "nb",
        }[contam_var]
        self.conv_max = conv_max
        self.fix_alpha = likelihood in ("nb_conv", "nb_mm")
        if self.fix_alpha:
            separate_retention_alpha = False
            stochastic_alpha = False
            component_alpha = False
        # retention_form picks how phi_cg (the estimated fraction of the cell's true
        # transcripts lost to outflow) becomes a retention factor r_cg:
        #   "exp"    r = exp(-alpha_g * phi)  -- the original soft form; alpha_g
        #            rescales how far to trust phi, and r stays positive for any phi.
        #   "linear" r = 1 - phi              -- phi at face value, the same "trust
        #            proseg's estimate" stance that fixing alpha = 1 takes for inflow.
        # Defaults follow that logic: "linear" wherever alpha is pinned at 1, "exp"
        # otherwise (which preserves nb_mean's existing behaviour exactly).
        if retention_form is None:
            retention_form = "linear" if self.fix_alpha else "exp"
        if retention_form not in ("exp", "linear"):
            raise ValueError(f"unknown retention_form {retention_form!r}")
        self.retention_form = retention_form
        if encoder_input not in ("raw", "decontaminated"):
            raise ValueError(f"unknown encoder_input {encoder_input!r}")
        self.encoder_input = encoder_input
        self.include_diffusion = include_diffusion
        # Independent ablation of the two diffusion terms (mirrors the toggles on
        # FactorizationVAE): include_retention gates the multiplicative outflow
        # correction exp(-alpha*phi), include_delta the additive inflow term
        # alpha*inflow. Both default on (the full model); either can be dropped to
        # attribute the diffusion effect to one mechanism. Only consulted when
        # include_diffusion is True.
        self.include_retention = include_retention
        self.include_delta = include_delta
        # component_alpha: fit a separate per-gene alpha for each proseg component
        # (its point-estimate expression-mixture assignment). Motivated by the
        # finding that contamination flagging is component-determined -- a single
        # per-gene alpha conflates the contamination level (which the NB likelihood
        # fits) with the near/far contrast (which DE needs), and those disagree for
        # low-signal genes; per-component alpha lets the correction adapt to the
        # component structure that actually drives proseg's flagging.
        self.component_alpha = component_alpha
        self.include_size_factor = include_size_factor
        self.rate_offset = rate_offset

        # prepare_encoder_input returns an [B, n_genes] tensor in BOTH branches --
        # the covariates are never concatenated -- so sizing this as
        # n_genes + n_covariates made include_size_factor=False raise a shape
        # error on the first batch (that path had never run). If a conditional
        # encoder is wanted, concatenate the design in prepare_encoder_input and
        # restore the wider input here.
        encoder_in_channels = n_genes
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

        self.stochastic_alpha = stochastic_alpha
        if include_diffusion and self.fix_alpha:
            # nb_conv: alpha is fixed at 1, so no leak parameter is trained.
            self.log_alpha = None
            self.log_alpha_logstd = None
            self.log_alpha_ret = None
        elif include_diffusion:
            # alpha_g: shared per-gene leak coefficient calibrating how far to trust
            # proseg's inflow/outflow estimates. alpha_g = 1 takes them at face value;
            # it's identifiable because contamination genes are under-fitted by
            # beta/z alone, while genuine DE genes already fit well and keep alpha
            # near 1. alpha = exp(log_alpha) >= 0 is unbounded above so alpha_g > 1
            # can absorb leakage proseg underestimates (a sigmoid bound to (0,1)
            # could not). Prior centered at alpha_g = 1, i.e. log_alpha = 0.
            # Shape is [n_components, n_genes] under component_alpha, else [n_genes].
            alpha_shape = (n_components, n_genes) if component_alpha else (n_genes,)
            self.log_alpha = nn.Parameter(torch.zeros(alpha_shape))

            # stochastic_alpha makes log_alpha a variational posterior N(log_alpha,
            # exp(log_alpha_logstd)^2) sampled during training, rather than a point
            # estimate. Its purpose is to propagate the contamination-vs-DE
            # identifiability uncertainty into beta: for a should-be-null gene near a
            # contaminating type, how much of the signal is alpha*inflow vs a real
            # beta effect is genuinely ambiguous, and a point alpha hands all that
            # residual confidence to beta -> overconfident, spurious DE calls once
            # there's enough power. This uncertainty is per-gene/systematic (it does
            # NOT average out over cells, unlike proseg's per-cell inflow variance),
            # so sampling alpha widens beta exactly where the split is unidentified.
            self.stochastic_alpha = stochastic_alpha
            if stochastic_alpha:
                self.log_alpha_logstd = nn.Parameter(torch.full(alpha_shape, -3.0))
            else:
                self.log_alpha_logstd = None

            # separate_retention_alpha decouples the outflow (retention) coefficient
            # from the inflow (delta) one. The paper shares a single alpha_g because
            # inflow and outflow are two views of one physical process, but a shared
            # coefficient forces one calibration to serve both terms; giving
            # retention its own log_alpha_ret lets each calibrate independently. When
            # shared (default), retention reuses log_alpha.
            if separate_retention_alpha and include_retention:
                self.log_alpha_ret = nn.Parameter(torch.zeros(n_genes))
            else:
                self.log_alpha_ret = None

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
        self,
        x_sub_tensor: torch.Tensor,
        batch_idx: torch.Tensor,
        inflow: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Normalize counts and compute size factor. Returns (dense encoder_in, log_sf).

        Accepts either a dense [B, n_genes] tensor or a sparse CSR batch; a sparse
        batch is densified here for the (dense) encoder. The sparse handle is not
        consumed by the encoder -- it is passed separately to nb_loss_sparse, which
        is the one place sparsity wins at these panel sizes.

        encoder_input="decontaminated" subtracts the inflow estimate from the counts
        before normalizing. The likelihood already knows which counts are
        contamination -- delta enters it as a separate stream -- but the ENCODER does
        not: it sees raw X, so a near-tumor macrophage full of leaked EPCAM looks to
        it like a cell that expresses EPCAM, and z_offset reconstructs it. Since the
        likelihood then wants that gene's own-signal rate to stay low, beta has to
        move to compensate, and because contamination is spatially structured the
        compensation is correlated with the design. Feeding the encoder X - delta
        removes the input that drives this, and tests whether the residual
        false-positive survives when z can no longer see the contamination.
        """
        x_dense = (
            x_sub_tensor.to_dense()
            if x_sub_tensor.layout == torch.sparse_csr
            else x_sub_tensor
        )
        if self.encoder_input == "decontaminated" and inflow is not None:
            inflow_dense = (
                inflow.to_dense() if inflow.layout == torch.sparse_csr else inflow
            )
            x_dense = (x_dense - inflow_dense).clamp(min=0.0)
        if self.include_size_factor:
            log_sf = self.log_sf_embed(batch_idx).squeeze(-1)
            x_norm = x_dense / (torch.exp(log_sf).unsqueeze(-1) + 1e-8) * 1000.0
            encoder_in = torch.log1p(x_norm)
        else:
            log_sf = None
            encoder_in = torch.log1p(x_dense)
        return encoder_in, log_sf

    def forward(
        self,
        x,
        covariates,
        inflow,
        phi,
        log_size_factor=None,
        component=None,
        inflow_var=None,
    ):
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

        delta = None
        contam_var = None
        if self.include_diffusion:
            # retention = exp(-alpha*phi) is 1 wherever phi=0, so mu is dense
            # regardless; densify the (sparse) inflow/phi batches here to build it.
            if self.fix_alpha:
                # nb_conv: alpha == 1, contamination taken at proseg's face value.
                alpha = 1.0
            else:
                if self.training and self.log_alpha_logstd is not None:
                    a_std = torch.exp(self.log_alpha_logstd.clamp(max=4.0))
                    log_alpha = self.log_alpha + torch.randn_like(a_std) * a_std
                else:
                    log_alpha = self.log_alpha
                if self.component_alpha:
                    # log_alpha is [n_components, n_genes]; gather this batch's rows.
                    assert component is not None
                    log_alpha = log_alpha[component]  # -> [B, n_genes]
                alpha = torch.exp(log_alpha.clamp(-10.0, 10.0))
            if self.include_retention:
                assert phi is not None
                if phi.layout == torch.sparse_csr:
                    phi = phi.to_dense()
                if self.retention_form == "linear":
                    retention = (1.0 - phi).clamp(min=1e-3)
                else:
                    alpha_ret = (
                        torch.exp(self.log_alpha_ret)
                        if self.log_alpha_ret is not None
                        else alpha
                    )
                    retention = torch.exp(-alpha_ret * phi)
                # Fold retention into the signal rate itself rather than only into
                # the combined mean: under nb_conv/nb_mm the signal stream is
                # A ~ NB(r*lam, psi), so the convolution must be handed r*lam. For
                # nb_mean this is equivalent -- mu is r*lam + delta either way.
                lam = retention * lam
            mu = lam
            if self.include_delta:
                assert inflow is not None
                if inflow.layout == torch.sparse_csr:
                    inflow = inflow.to_dense()
                delta = alpha * inflow
                mu = mu + delta
                if self.contam_family != "poisson":
                    assert inflow_var is not None, (
                        f"contam_var={self.contam_var!r} needs proseg's "
                        "expected_inflow_var layer"
                    )
                    if inflow_var.layout == torch.sparse_csr:
                        inflow_var = inflow_var.to_dense()
                    # alpha == 1 whenever a non-Poisson contamination arm is in use
                    # (fix_alpha), so V_f needs no rescaling here.
                    contam_var = (
                        inflow_var
                        if self.contam_family == "binomial"
                        else delta + inflow_var
                    )
            mu = mu + self.rate_offset
        else:
            mu = lam + self.rate_offset

        # lam (own-signal rate) and delta (contamination rate) are returned
        # separately -- the nb_conv/nb_mm likelihoods keep the two streams distinct
        # rather than using their sum mu. contam_var is the target variance of the
        # contamination count (None means "Poisson", i.e. equal to delta). In nb_mean
        # mode lam excludes the rate_offset that mu carries; the offset is a numerical
        # floor for the combined-mean NB and is not part of the generative signal rate.
        return mu, lam, delta, contam_var, z_mu, z_logstd, beta
