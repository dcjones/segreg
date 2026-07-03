import math
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from anndata import AnnData
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from spatialdata import SpatialData
from tqdm import tqdm

from .data import SparseBatchSampler, estimate_phi, load_proseg_data
from .losses import (
    alpha_log_prior_loss,
    dirichlet_purity_loss,
    gene_floor_loss,
    metagene_entropy_loss,
    min_volume_loss,
)


def sparsemax(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Sparsemax projection onto the simplex (Martins & Astudillo, 2016).

    Differentiable like softmax, but -- unlike softmax, which assigns strictly
    positive weight to every entry -- can assign exact zero weight to entries
    below a data-dependent threshold. Used as an alternative to softmax for
    FactorizationVAE.v_norm(): softmax guarantees every gene gets nonzero weight
    in every factor, so a rare cell type's marker gene always leaks a little
    reconstructed signal into unrelated factors regardless of how much rank the
    model has (see DIFFUSION_INVESTIGATION_NOTES.md sec. 16-17, "manufactured
    positives" -- more factors gave only partial relief because this leakage is
    structural to softmax, not a capacity problem). Ordinary autograd through
    sort/cumsum/gather/clamp already reproduces sparsemax's known analytic
    Jacobian (the threshold/support-size term has zero gradient almost
    everywhere, which autograd handles for free since it comes from a boolean
    comparison), so no custom backward is needed.
    """
    z = z - z.max(dim=dim, keepdim=True).values
    z_sorted, _ = torch.sort(z, dim=dim, descending=True)
    n = z.shape[dim]
    rho = torch.arange(1, n + 1, device=z.device, dtype=z.dtype)
    shape = [1] * z.dim()
    shape[dim] = n
    rho = rho.view(shape)
    cumsum = z_sorted.cumsum(dim=dim)
    support = (1 + rho * z_sorted) > cumsum
    k = support.sum(dim=dim, keepdim=True).to(z.dtype)
    cumsum_k = torch.gather(cumsum, dim, (k.long() - 1).clamp(min=0))
    tau = (cumsum_k - 1) / k
    return torch.clamp(z - tau, min=0)


def entmax15(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """1.5-entmax: Tsallis alpha=1.5 sparse projection onto the simplex
    (Peters, Niculae & Martins, 2019). Interpolates between softmax (alpha=1,
    dense, strictly positive everywhere) and sparsemax (alpha=2, can hit exact
    zero but has a hard, constant-slope cutoff). Tried as the follow-up to
    sparsemax (DIFFUSION_INVESTIGATION_NOTES.md sec. 18): sparsemax collapsed
    T-cell/B-cell marker reconstruction to literal zero everywhere, traced to
    coordinates getting prematurely zeroed early in training with no way back
    (a coordinate excluded from sparsemax's support has strictly zero local
    gradient, and sparsemax's support tends to be aggressively small even for
    modest logit spread). entmax15's solution has the same clamped-quadratic
    form p_i = [z_i - tau]_+^2 rather than sparsemax's clamped-linear
    p_i = [z_i - tau]_+ -- for the same logits it yields a strictly larger
    support set (less aggressive sparsification), so weak/rare signals are less
    likely to be prematurely excluded before they've established themselves in
    some factor, while still reaching exact zero once logits genuinely diverge.

    Closed-form solution (no bisection needed for alpha=1.5): after shifting
    for numerical stability and dividing by (alpha-1)=0.5, the optimal tau is
    found by scanning candidate support sizes in sorted order, using the
    running mean and mean-of-squares of the top-k logits to solve the
    per-k threshold in closed form, then picking the largest self-consistent k
    (mirrors sparsemax's sort-based algorithm; only the threshold formula
    differs). As with sparsemax, plain autograd through this reproduces the
    correct Jacobian since the support-size selection has no gradient path.
    """
    z = z - z.max(dim=dim, keepdim=True).values
    z = z / 2
    z_sorted, _ = torch.sort(z, dim=dim, descending=True)
    n = z.shape[dim]
    rho = torch.arange(1, n + 1, device=z.device, dtype=z.dtype)
    shape = [1] * z.dim()
    shape[dim] = n
    rho = rho.view(shape)

    mean = z_sorted.cumsum(dim=dim) / rho
    mean_sq = (z_sorted**2).cumsum(dim=dim) / rho
    ss = rho * (mean_sq - mean**2)
    # sqrt's gradient is unbounded as its argument -> 0+ (verified: exactly inf
    # at delta==0, ~5e9 at delta==1e-20) -- a legitimate, not-just-floating-point
    # boundary case that occurs often enough with per-cell entmax15 calls (e.g.
    # composition_activation="entmax15" in SparseCompositionAbundanceEncoder,
    # ~1024 calls/batch vs. v_norm()'s ~50/batch) to eventually corrupt training
    # via an inf gradient propagating through clip_grad_norm_'s total-norm
    # computation (grad_clip scales by 1/total_norm, which becomes 0 when
    # total_norm is inf, and 0*inf=NaN corrupts every parameter with any
    # nonzero gradient that step). 1e-6 caps the worst-case gradient at ~500,
    # comfortably within grad_clip's normal operating range, with negligible
    # effect on the forward value.
    delta = torch.clamp((1 - ss) / rho, min=1e-6)
    tau_candidates = mean - torch.sqrt(delta)

    support = tau_candidates <= z_sorted
    support_size = support.sum(dim=dim, keepdim=True)
    tau_star = torch.gather(tau_candidates, dim, (support_size - 1).clamp(min=0))

    return torch.clamp(z - tau_star, min=0) ** 2


warnings.filterwarnings(
    "ignore",
    message="Sparse CSR tensor support is in beta state",
    category=UserWarning,
)


def _nndsvd_h_init(X: csr_matrix, k: int, init_ncells: int = 10000) -> np.ndarray:
    """Compute NNDSVD-based initialization for the H (gene programs) matrix.

    Runs truncated SVD, extracts non-negative factors via the NNDSVD procedure,
    and returns log(H_norm) where H_norm is the row-normalized H factor.
    This is the inverse of the softplus activation applied during the forward pass.
    """
    m, n = int(X.shape[0]), int(X.shape[1])  # type: ignore[index]
    if m > init_ncells:
        rng = np.random.default_rng(42)
        indices = rng.choice(m, init_ncells, replace=False)
        indices.sort()
        X_init = X[indices]
    else:
        X_init = X

    U, S, Vt = svds(X_init.astype(np.float64), k=k)  # type: ignore[misc]
    idx = np.argsort(S)[::-1]
    S = S[idx]
    U = U[:, idx]  # type: ignore[index]
    Vt = Vt[idx, :]  # type: ignore[index]
    H = np.zeros((k, n))
    for j in range(k):
        x = U[:, j]
        y = Vt[j, :]
        xp = np.maximum(x, 0)
        xn = np.abs(np.minimum(x, 0))
        yp = np.maximum(y, 0)
        yn = np.abs(np.minimum(y, 0))
        xpn = np.linalg.norm(xp)
        xnn = np.linalg.norm(xn)
        ypn = np.linalg.norm(yp)
        ynn = np.linalg.norm(yn)
        mp = xpn * ypn
        mn = xnn * ynn
        if mp > mn:
            H[j, :] = yp * xpn * np.sqrt(S[j])
        else:
            H[j, :] = yn * xnn * np.sqrt(S[j])

    H = H + 1e-6
    H_norm = H / H.sum(axis=1, keepdims=True)
    return np.log(H_norm).astype(np.float32)


def _kmeans_component_init(
    X: csr_matrix,
    v_init: np.ndarray,
    n_components: int,
    mixture_space: str,
    init_ncells: int = 10000,
) -> tuple[np.ndarray, np.ndarray]:
    """Data-driven seed for the latent GMM components.

    The default random seed (randn·0.1 near the origin) bears no relation to
    where cells actually sit in the mixture space, which can trap the clustering
    in a bad basin. Here we build a cheap proxy for the cell loadings by
    projecting a sample of cells onto the NNDSVD metagenes (W ≈ X·Hᵀ), map it
    into the same space the GMM lives in (CLR for "clr", raw for "u"), and
    k-means it. Centroids seed component_means; within-cluster variances seed
    component_log_vars so components start appropriately sized rather than all at
    σ²=1. This is only an initialization -- the projection need not match the
    trained encoder exactly, just place the components near real data structure.
    """
    from sklearn.cluster import KMeans

    m = int(X.shape[0])  # type: ignore[arg-type]
    rng = np.random.default_rng(42)
    if m > init_ncells:
        idx = np.sort(rng.choice(m, init_ncells, replace=False))
        X_s = X[idx]
    else:
        X_s = X
    # v_init = log(H_norm); H_norm rows sum to 1. W ≈ X·Hᵀ is a rough per-cell
    # loading (nonnegative), enough to locate cluster centers.
    H_norm = np.exp(v_init)  # [k, n]
    W = np.asarray(X_s @ H_norm.T, dtype=np.float64)  # [n_sample, k]
    W = np.maximum(W, 0.0)

    if mixture_space == "clr":
        log_w = np.log(np.clip(W, 1e-3, None))
        feats = log_w - log_w.mean(axis=1, keepdims=True)
    else:
        feats = W

    km = KMeans(n_clusters=n_components, n_init=10, random_state=42).fit(feats)
    means = km.cluster_centers_.astype(np.float32)  # [n_components, k]
    log_vars = np.zeros((n_components, feats.shape[1]), dtype=np.float32)
    for c in range(n_components):
        sel = km.labels_ == c
        if sel.sum() > 1:
            var = feats[sel].var(axis=0) + 1e-4
            log_vars[c] = np.log(var).astype(np.float32)
    return means, log_vars


class SparseLinear(nn.Module):
    """
    Linear layer that handles both sparse CSR and dense input.
    Weight is stored as [in_features, out_features] for torch.sparse.mm compatibility.
    Initialized with LeCun normal (std = sqrt(1/fan_in)).
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        std = math.sqrt(1.0 / in_features)
        nn.init.normal_(self.weight, 0.0, std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.layout == torch.sparse_csr:
            return torch.sparse.mm(x, self.weight) + self.bias
        return x @ self.weight + self.bias


class SparseFactorizationEncoder(nn.Module):
    """
    Maps a sparse count matrix into metagene rates. When use_diffusion_input is
    True, also takes inflow/phi as additional sparse inputs, each through their
    own SparseLinear branch summed into the same pre-activation, so the encoder
    can learn to discount contamination on a per-cell basis instead of only
    seeing raw (contaminated) counts. The inflow/phi branches are zero-initialized
    so the encoder starts out identical to the X-only baseline.
    """

    def __init__(
        self, n: int, k: int, use_diffusion_input: bool = False,
        diffusion_input_mode: str = "additive",
    ):
        super().__init__()
        self.layer = SparseLinear(n, k)
        self.use_diffusion_input = use_diffusion_input
        self.diffusion_input_mode = diffusion_input_mode
        if use_diffusion_input:
            self.inflow_layer = SparseLinear(n, k)
            self.phi_layer = SparseLinear(n, k)
            nn.init.zeros_(self.inflow_layer.weight)
            nn.init.zeros_(self.inflow_layer.bias)
            nn.init.zeros_(self.phi_layer.weight)
            nn.init.zeros_(self.phi_layer.bias)

    def forward(
        self,
        X: torch.Tensor,
        inflow: torch.Tensor | None = None,
        φ: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.layer(X)
        if self.use_diffusion_input:
            assert inflow is not None and φ is not None
            if self.diffusion_input_mode == "subtractive":
                # Inflow (contamination) can only DISCOUNT factor loadings, never
                # inflate them -- relu(·)>=0 subtracted, zero-init so it starts at 0
                # (baseline). This is the inductive bias the free additive form
                # lacked: on the honest metrics the additive encoder "leaned into"
                # inflow as a generic feature and over-removed signal. Outflow φ
                # stays free-additive (it should *recover* signal lost to outflow).
                h = h - F.relu(self.inflow_layer(inflow)) + self.phi_layer(φ)
            else:
                h = h + self.inflow_layer(inflow) + self.phi_layer(φ)
        return F.softplus(h)


def log1p_sparse(X: torch.Tensor) -> torch.Tensor:
    """torch.log1p for a sparse_csr_tensor, without densifying.

    log1p(0) = 0, so applying it to just the nonzero values preserves the
    sparsity pattern exactly -- no need to materialize the dense zeros.
    """
    if X.layout != torch.sparse_csr:
        return torch.log1p(X)
    return torch.sparse_csr_tensor(
        X.crow_indices(),
        X.col_indices(),
        torch.log1p(X.values()),
        size=X.shape,
        dtype=X.dtype,
        device=X.device,
    )


class SparseCompositionAbundanceEncoder(nn.Module):
    """Alternative to SparseFactorizationEncoder: splits u into a per-cell
    abundance *scale* (a scalar) and a metagene *composition* (a simplex
    vector over factors), learned by separate linear readouts, instead of
    mixing "how much" and "which factors" into one unconstrained softplus
    output as SparseFactorizationEncoder does.

    Motivation (DIFFUSION_INVESTIGATION_NOTES.md sec 22): entangling magnitude
    and composition in a single softplus output means the encoder has to
    simultaneously get a cell's total signal strength right *and* its factor
    mixture right through the same gate, with the (typically large-magnitude)
    abundance signal potentially dominating gradient contributions to
    low-loading factors that matter for rare cell types. Separating them
    means the composition head only ever has to answer "given this cell's
    shape (not scale), which factors does it look like" -- and lets us try
    sparsifying activations (sparsemax/entmax15) on composition specifically,
    independent of the (always-positive, never-zero) abundance scale.

    composition_layer sees log1p(X) (compresses dynamic range so a few
    highly-expressed genes don't dominate the *shape* signal); scale_layer
    sees raw X (total magnitude is exactly what it's meant to capture).
    """

    def __init__(self, n: int, k: int, composition_activation: str = "softmax"):
        super().__init__()
        if composition_activation not in ("softmax", "sparsemax", "entmax15"):
            raise ValueError(
                f"Unknown composition_activation: {composition_activation}. "
                f"Must be one of: 'softmax', 'sparsemax', 'entmax15'."
            )
        self.composition_activation = composition_activation
        self.composition_layer = SparseLinear(n, k)
        self.scale_layer = SparseLinear(n, 1)
        # Populated by forward(), read by loss_fn for dirichlet_purity_loss --
        # composition (not u, which also carries the abundance scale) is what a
        # Dirichlet(alpha<1) purity prior needs to be computed against.
        self.last_composition: torch.Tensor | None = None

    def forward(self, X: torch.Tensor, X_log1p: torch.Tensor) -> torch.Tensor:
        comp_logits = self.composition_layer(X_log1p)
        if self.composition_activation == "sparsemax":
            composition = sparsemax(comp_logits, dim=1)
        elif self.composition_activation == "entmax15":
            composition = entmax15(comp_logits, dim=1)
        else:
            composition = F.softmax(comp_logits, dim=1)
        self.last_composition = composition
        scale = F.softplus(self.scale_layer(X)).squeeze(-1)
        return scale.unsqueeze(-1) * composition


class FactorizationVAE(nn.Module):
    def __init__(
        self,
        n: int,
        k: int,
        v_init: np.ndarray | None,
        include_diffusion: bool,
        diffusion_aware_encoder: bool = False,
        decontaminate_encoder: bool = False,
        diffusion_input_mode: str = "subtractive",
        per_cell_alpha: bool = False,
        include_retention: bool = True,
        include_delta: bool = True,
        metagene_activation: str = "softmax",
        encoder_architecture: str = "joint",
        composition_activation: str = "softmax",
        gene_expression: torch.Tensor | None = None,
        n_components: int | None = None,
        resp_temperature: float = 2.0,
        likelihood: str = "poisson",
        mixture_space: str = "clr",
        mixture_prior: str = "generative",
        component_means_init: np.ndarray | None = None,
        component_log_vars_init: np.ndarray | None = None,
    ):
        super().__init__()
        self.resp_temperature = resp_temperature
        if likelihood not in ("poisson", "nb"):
            raise ValueError(
                f"Unknown likelihood: {likelihood}. Must be 'poisson' or 'nb'."
            )
        self.likelihood = likelihood
        # Space the latent GMM operates in. "u" = the raw loadings scale·composition
        # (the original; clusters partly on cell size, per the scale-confound
        # diagnostic). "clr" = centered-log-ratio of u, i.e. a logistic-normal
        # mixture in Aitchison geometry. CLR is scale-invariant (clr(s·c)=clr(c)),
        # so it removes the cell-size axis exactly and clusters on composition
        # shape alone, while reusing the same Gaussian components/responsibilities.
        if mixture_space not in ("u", "clr"):
            raise ValueError(
                f"Unknown mixture_space: {mixture_space}. Must be 'u' or 'clr'."
            )
        self.mixture_space = mixture_space

        if encoder_architecture not in ("joint", "composition_abundance"):
            raise ValueError(
                f"Unknown encoder_architecture: {encoder_architecture}. "
                f"Must be one of: 'joint', 'composition_abundance'."
            )
        if encoder_architecture == "composition_abundance" and (
            diffusion_aware_encoder or decontaminate_encoder
        ):
            raise ValueError(
                "encoder_architecture='composition_abundance' is not yet "
                "supported together with diffusion_aware_encoder or "
                "decontaminate_encoder."
            )
        self.encoder_architecture = encoder_architecture

        if metagene_activation not in ("softmax", "sparsemax", "entmax15"):
            raise ValueError(
                f"Unknown metagene_activation: {metagene_activation}. "
                f"Must be one of: 'softmax', 'sparsemax', 'entmax15'."
            )
        self.metagene_activation = metagene_activation

        # "regularizer": the GMM is a soft density term bolted onto the
        # deterministic point-estimate u (the original). "generative": a proper
        # VaDE-style mixture *prior* -- z is sampled from q(z|x)=N(clr(u), σ²),
        # the composition is decoded from the *sampled* z (u_z = scale·softmax(z)),
        # and responsibilities are evaluated at that same z, so component
        # assignment is coupled to reconstruction likelihood (a dedicated
        # component is rewarded for explaining a cell's counts). This is the
        # structural analog of proseg's generative Gamma mixture, motivated by the
        # probe finding that immune types are linearly separable in the latent yet
        # the density-only regularizer refuses to carve them out.
        if mixture_prior not in ("regularizer", "generative"):
            raise ValueError(
                f"Unknown mixture_prior: {mixture_prior}. "
                f"Must be 'regularizer' or 'generative'."
            )
        if mixture_prior == "generative" and mixture_space != "clr":
            raise ValueError(
                "mixture_prior='generative' requires mixture_space='clr' "
                "(it decodes the composition as softmax(z) in the CLR latent)."
            )
        self.mixture_prior = mixture_prior
        if mixture_prior == "generative":
            # Global (cell-independent) encoder log-variance for q(z|x). A per-cell
            # amortized head was tried and was a wash-to-slightly-worse (no gain on
            # homogeneity/recall, worse inter_jac, and it did NOT reduce the
            # run-to-run resolution spread it was meant to), so the shared σ²
            # stays. Learns to ~σ²≈1 in practice.
            self.encoder_log_var = nn.Parameter(torch.full((k,), math.log(0.1)))

        if v_init is None:
            v = torch.empty(k, n)
            nn.init.normal_(v, std=1.0 / math.sqrt(n))
            self.v = nn.Parameter(v)
        else:
            self.v = nn.Parameter(torch.from_numpy(v_init))

        # Per-gene NB dispersion r_g = exp(log_r); variance = μ + μ²/r. Larger r →
        # Poisson limit. Only used when likelihood="nb". Initialized at r=10
        # (mild overdispersion) and learned per gene; the point (per the
        # scale-confound diagnostic) is to let r shrink for overdispersed
        # high-count structural genes, softening their grip on the reconstruction
        # gradient so low-count marker genes can shape the latent.
        if likelihood == "nb":
            self.log_r = nn.Parameter(torch.full((n,), math.log(10.0)))

        self.include_diffusion = include_diffusion
        self.diffusion_aware_encoder = diffusion_aware_encoder and include_diffusion
        # Deterministic alternative to diffusion_aware_encoder: instead of letting the
        # encoder learn (from scratch, with no bias toward cancellation) whether/how to
        # use raw inflow as an input -- which empirically made things worse, since
        # inflow is itself just another feature correlated with "near a contaminating
        # neighbor" that the encoder is free to lean into rather than subtract out --
        # explicitly discount the already-fit alpha_g * inflow_cg from X before encoding.
        # Uses a multiplicative, scale-invariant discount (mirroring retention's
        # exp(-alpha*phi)) rather than absolute subtraction: relu(X - alpha*inflow)
        # trivially zeroes out low-count genes whenever the inflow estimate is
        # comparable to X in absolute terms, even for a small, non-pathological
        # fraction of contamination -- confirmed via a pairwise-Jaccard check across
        # six cell types, where absolute subtraction badly degraded separation among
        # rarer, lower-count cell types (B-cell, mast, endothelial) while only helping
        # the high-count tumor axis.
        self.decontaminate_encoder = decontaminate_encoder and include_diffusion
        # Ablation: the paper ties retention (outflow, multiplicative) and delta
        # (inflow, additive) to the same alpha_g on physical grounds (see
        # segreg-paper.typ lines ~140-161), but retention's Poisson-NLL gradient is
        # scaled by rho = exp(-alpha*phi), which dampens training signal exactly at
        # cell-type boundaries where phi is high -- see DIFFUSION_INVESTIGATION_NOTES.md
        # sec. 9. include_retention=False drops the multiplicative outflow term
        # entirely (mu = lambda + delta), keeping only the additive inflow
        # correction, to test whether outflow/phi is the source of that damage.
        # Turned out not to be (sec. 11) -- include_delta=False is the complementary
        # ablation, dropping the additive inflow term instead (mu = retention*lambda),
        # to isolate whether delta/inflow is the real driver (sec. 12).
        self.include_retention = include_retention
        self.include_delta = include_delta
        if encoder_architecture == "composition_abundance":
            self.encoder = SparseCompositionAbundanceEncoder(
                n, k, composition_activation=composition_activation
            )
        else:
            self.encoder = SparseFactorizationEncoder(
                n, k, use_diffusion_input=self.diffusion_aware_encoder,
                diffusion_input_mode=diffusion_input_mode,
            )
        if include_diffusion:
            # alpha = exp(log_alpha) >= 0, unbounded above: the paper's model (see
            # segreg-paper.typ) allows alpha_g > 1 to accommodate leakage that Proseg
            # underestimates, with a prior centered at alpha_g = 1 (log_alpha = 0),
            # not at alpha_g = 0. A sigmoid transform (bounded to (0,1)) can't express
            # "Proseg underestimated this gene" at all.
            self.log_α = nn.Parameter(torch.zeros(n))
            # Per-gene mean expression, used to weight the alpha prior so low-count
            # genes (where there's little evidence to justify correction) stay
            # anchored near alpha=1 while high-count genes get more freedom -- see
            # alpha_log_prior_loss. Registered as a buffer so it moves with .to(device).
            if gene_expression is not None:
                self.register_buffer("gene_expression", gene_expression)
            else:
                self.gene_expression = None

            # Per-cell alpha: α_cg = α_g · exp(head(inflow_c) + head(φ_c)). The
            # global α_g removes the same inflow fraction from every cell, which is
            # too blunt -- a tumor-adjacent immune cell has more real contamination
            # than an interior one, and a single α_g can't be both ≈1 (interior)
            # and >1 (boundary). The per-cell modulation reads the cell's own
            # inflow/φ (the signal for how contaminated it is) and scales α up/down.
            # Gene-specificity still comes from inflow_cg (data); the head supplies
            # the cell-level factor. Zero-init -> modulation ≡ 1, so training starts
            # at the global-α model and learns the per-cell adjustment.
            self.per_cell_alpha = per_cell_alpha and include_diffusion
            if self.per_cell_alpha:
                self.alpha_inflow_head = SparseLinear(n, 1)
                self.alpha_phi_head = SparseLinear(n, 1)
                for h in (self.alpha_inflow_head, self.alpha_phi_head):
                    nn.init.zeros_(h.weight)
                    nn.init.zeros_(h.bias)
        else:
            self.per_cell_alpha = False

        # GMM prior on U (VaDE/GMVAE-style). Cluster responsibilities are computed
        # *analytically* from the Gaussian components (see compute_gmm_loss), not by
        # a separate amortized MLP head. An earlier version used an MLP head to
        # predict mixture logits from u; it was a second, redundant parametrization
        # of cluster membership decoupled from the actual component geometry, and it
        # drifted between regimes run-to-run (identical config/seed swung the argmax
        # from a clean 10-way split to a 2-way collapse, with matching swings in the
        # marker metrics). Tying responsibilities to the components directly removes
        # that instability and the extra parameters.
        self.n_components = n_components
        if n_components is not None:
            # Component means in the mixture space (CLR or u). Seeded from k-means
            # on projected cell loadings when a data-driven init is supplied (see
            # _kmeans_component_init), else a random blob near the origin.
            if component_means_init is not None:
                self.component_means = nn.Parameter(
                    torch.from_numpy(component_means_init.astype(np.float32))
                )
            else:
                self.component_means = nn.Parameter(torch.randn(n_components, k) * 0.1)
            # Component log-variances (learnable). Seeded from within-cluster
            # variance when a data-driven init is supplied, else log(1)=0 (σ²=1,
            # matching the var_reg prior's center so components start
            # well-conditioned rather than as narrow spikes).
            if component_log_vars_init is not None:
                self.component_log_vars = nn.Parameter(
                    torch.from_numpy(component_log_vars_init.astype(np.float32))
                )
            else:
                self.component_log_vars = nn.Parameter(torch.zeros(n_components, k))
            # Global log mixing weights π_k. Fixed uniform (a buffer, not a
            # Parameter): the load-balancing term already governs aggregate usage,
            # and leaving the mixing weights learnable just gives collapse another
            # unconstrained degree of freedom to exploit.
            self.register_buffer("mixture_logits", torch.zeros(n_components))

    def v_norm(self) -> torch.Tensor:
        if self.metagene_activation == "sparsemax":
            return sparsemax(self.v, dim=1)
        if self.metagene_activation == "entmax15":
            return entmax15(self.v, dim=1)
        return F.softmax(self.v, dim=1)

    def alpha(
        self, inflow: torch.Tensor | None = None, φ: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Per-gene α_g = exp(log_α), shape [n]; or, if per_cell_alpha, the per-cell
        α_cg = α_g · exp(head(inflow_c) + head(φ_c)), shape [B, n]. Both broadcast
        against [B, n] inflow/φ in the reconstruction. Shared by forward() and
        get_corrected_expression() so they stay consistent."""
        a = torch.exp(self.log_α)  # [n]
        if self.per_cell_alpha and inflow is not None:
            logit = self.alpha_inflow_head(inflow)  # [B, 1]
            if φ is not None:
                logit = logit + self.alpha_phi_head(φ)
            return a.unsqueeze(0) * torch.exp(logit)  # [B, n]
        return a

    def encode(
        self,
        X: torch.Tensor,
        inflow: torch.Tensor | None = None,
        φ: torch.Tensor | None = None,
        α: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Computes the per-cell metagene loadings u, using whichever encoder-input
        strategy this model was configured with. Shared by forward() and the
        FactorizationModel inference methods so they can't drift out of sync."""
        if self.decontaminate_encoder:
            assert inflow is not None and α is not None
            X_dense = X.to_dense() if X.layout == torch.sparse_csr else X
            inflow_dense = (
                inflow.to_dense() if inflow.layout == torch.sparse_csr else inflow
            )
            inflow_frac = inflow_dense / (X_dense + 1e-6)
            encoder_input = X_dense * torch.exp(-α.unsqueeze(0) * inflow_frac)
            return self.encoder(encoder_input)
        elif self.diffusion_aware_encoder:
            return self.encoder(X, inflow, φ)
        elif self.encoder_architecture == "composition_abundance":
            return self.encoder(X, log1p_sparse(X))
        else:
            return self.encoder(X)

    def _mixture_features(self, u: torch.Tensor) -> torch.Tensor:
        """Project the loadings u into the space the latent GMM clusters in.

        "u": identity (cluster scale·composition directly).
        "clr": centered log-ratio, clr(u)_i = log u_i - mean_j log u_j. Because
        clr(s·c) = clr(c), this discards the per-cell scale exactly and leaves
        only the composition shape, mapped to the zero-sum subspace of R^k where
        a Gaussian mixture (a logistic-normal mixture on the simplex) is the
        appropriate model. A small floor guards log(0); u from the joint encoder
        is a strictly-positive softplus, so the floor only bounds tiny entries.
        """
        if self.mixture_space == "clr":
            log_u = torch.log(u.clamp_min(1e-3))
            return log_u - log_u.mean(dim=-1, keepdim=True)
        return u

    def compute_gmm_loss(
        self, u: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Compute the per-cell GMM prior log-density on u and the analytic
        cluster responsibilities (GMM posterior over components).

        Returns:
            log_mixture_prob: [batch] per-cell log Σ_k π_k N(u | μ_k, σ²_k), or
                None if n_components is None. Returned per-cell (not reduced) so
                loss_fn can *sum* it over the batch -- the reconstruction term is
                a batch sum (poisson_logprob_sparse), so a .mean() here would make
                the prior ~batch_size× too weak to actually shape u (it would only
                fit the components to a latent space reconstruction alone
                organizes). See loss_fn.
            γ: [batch, n_components] responsibilities γ_ik = p(k | u_i), computed
                analytically from the components (softmax of the per-component log
                joint), or None. This replaces the old amortized MLP head: it is a
                deterministic function of u and the Gaussians, so it cannot drift
                into a different clustering regime the way the decoupled head did.
        """
        if self.n_components is None:
            return None, None

        # Map the loadings into the space the mixture lives in. For "clr" this
        # strips the cell-size axis (CLR is scale-invariant) so the Gaussian
        # components fit composition *shape* rather than scale·shape.
        z = self._mixture_features(u)  # [B, k]
        return self._gmm_terms(z, tempered=self.mixture_prior != "generative")

    def _gmm_terms(
        self, z: torch.Tensor, tempered: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """GMM marginal log-density and responsibilities at latent z (already in
        the mixture space). Split out from compute_gmm_loss so the generative path
        can evaluate it at a *sampled* z rather than at clr(u).

        tempered=True divides the responsibility logits by resp_temperature·k --
        needed by the regularizer path to keep the load-balancing gradient alive
        at high k (an untempered posterior saturates to one-hot). The generative
        (VaDE) path passes tempered=False: its ELBO requires the *true* posterior
        γ=p(c|z), and it relies on separated k-means-initialized components rather
        than a balance term to avoid collapse."""
        # log N(z | μ_k, σ²_k) = -0.5 [ Σ_g log σ²_kg + Σ_g (z_g - μ_kg)²/σ²_kg ]
        # (dropping the z-independent -0.5*d*log(2π) constant), vectorized over
        # components.
        var = torch.exp(self.component_log_vars) + 1e-6  # [K, k]
        diff = z.unsqueeze(1) - self.component_means.unsqueeze(0)  # [B, K, k]
        mahal = (diff**2 / var.unsqueeze(0)).sum(dim=-1)  # [B, K]
        log_norm = self.component_log_vars.sum(dim=-1).unsqueeze(0)  # [1, K]
        log_probs = -0.5 * (log_norm + mahal)  # [B, K]

        # log joint p(u, k) = log N(u | k) + log π_k, up to the shared 2π constant.
        log_pi = F.log_softmax(self.mixture_logits, dim=-1)  # [K]
        log_joint = log_probs + log_pi.unsqueeze(0)  # [B, K]

        # Marginal log p(u) = logsumexp_k log joint. Used as-is (temperature 1) for
        # the generative NLL, which must stay the exact GMM marginal.
        log_mixture_prob = torch.logsumexp(log_joint, dim=-1)  # [B]

        # Responsibilities for the clustering *regularizers* (entropy/balance) and
        # for reported assignments are tempered. In a k-dim latent the untempered
        # posterior saturates to one-hot (the Mahalanobis term is a sum over k dims,
        # so between-component log-density gaps grow ~linearly in k -- hundreds at
        # k=100), and a saturated softmax has ~zero gradient, which silently kills
        # the load-balancing term and lets usage collapse back onto 1-2 components
        # (empirically: two components held 95% of cells at T=1). We divide the
        # logits by resp_temperature * k: the extra factor of k normalizes out the
        # dimension so resp_temperature is an O(1), latent-dim-independent knob
        # (~2 keeps all components alive with well-spread usage here). This keeps
        # the balance gradient alive while still tracking the component geometry, so
        # assignments stay stable run-to-run -- unlike the old decoupled MLP head.
        if tempered:
            k_dim = self.component_means.shape[1]
            γ = F.softmax(log_joint / (self.resp_temperature * k_dim), dim=-1)  # [B, K]
        else:
            γ = F.softmax(log_joint, dim=-1)  # true posterior p(c|z) for the ELBO

        return log_mixture_prob, γ

    def forward(
        self, X: torch.Tensor, inflow: torch.Tensor | None, φ: torch.Tensor | None
    ):
        # Per-gene α for the encoder's contamination discount (decontaminate_encoder);
        # the reconstruction below uses alpha() which may be per-cell.
        α_enc = torch.exp(self.log_α) if self.include_diffusion else None
        u = self.encode(X, inflow, φ, α_enc)

        aux = None
        if self.mixture_prior == "generative" and self.n_components is not None:
            # z ~ q(z|x) = N(z_mean, σ²) with z_mean = clr(u); decode the
            # composition from the *sampled* z so reconstruction and cluster
            # responsibilities share the latent. At eval (no grad / not training)
            # z = z_mean, so u_z = scale·softmax(clr(u)) = u exactly and the
            # inference/eval paths are unchanged.
            z_mean = self._mixture_features(u)  # clr(u), [B, k]
            if self.training:
                std = torch.exp(0.5 * self.encoder_log_var)
                z = z_mean + std * torch.randn_like(z_mean)
            else:
                z = z_mean
            scale = u.sum(dim=-1, keepdim=True)
            u = scale * F.softmax(z, dim=-1)  # u_z drives the decoder
            log_mixture_prob, π = self._gmm_terms(z, tempered=False)
            aux = (z_mean, self.encoder_log_var)
        else:
            # Per-cell GMM prior log-density and amortized responsibilities.
            log_mixture_prob, π = self.compute_gmm_loss(u)

        λ = u @ self.v_norm()

        if self.include_diffusion:
            assert inflow is not None and φ is not None
            α = self.alpha(inflow, φ)  # [n] or [B, n]
            δ = α * inflow.to_dense() if self.include_delta else 0.0
            if self.include_retention:
                retention = torch.exp(-α * φ.to_dense())
                λ = retention * λ + δ
            else:
                λ = λ + δ

        return u, λ, log_mixture_prob, π, aux

    def metagene_regularization(self):
        pass


def _sparse_row_col_indices(
    X: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (row_idx, col_idx) arrays indexing every non-zero element in a sparse_csr_tensor."""
    crow = X.crow_indices()
    col_idx = X.col_indices()
    batch_size = X.shape[0]
    row_idx = torch.repeat_interleave(
        torch.arange(batch_size, device=X.device),
        crow[1:] - crow[:-1],
    )
    return row_idx, col_idx


def loss_fn(
    model: FactorizationVAE,
    X: torch.Tensor,
    inflow: torch.Tensor | None = None,
    φ: torch.Tensor | None = None,
    row_idx: torch.Tensor | None = None,
    alpha_reg: float = 0.0,
    sparsity_reg: float = 0.0,
    metagene_reg_strength: float = 0.0,
    gene_floor_reg: float = 0.0,
    gene_floor_margin: float = -3.0,
    dirichlet_reg: float = 0.0,
    dirichlet_alpha: float = 0.5,
    mixture_strength: float = 0.0,
    entropy_weight: float = 0.01,
    balance_weight: float = 1.0,
    var_reg: float = 1.0,
) -> torch.Tensor:
    u, λ, log_mixture_prob, π, aux = model(X, inflow, φ)
    v = model.v_norm()

    if model.likelihood == "nb":
        loss = -nb_logprob_sparse(
            λ, X, model.log_r, constant_terms=False, row_idx=row_idx
        )
    else:
        loss = -poisson_logprob_sparse(λ, X, constant_terms=False, row_idx=row_idx)
    if model.include_diffusion:
        loss = loss + alpha_log_prior_loss(
            model.log_α, alpha_reg, gene_expression=model.gene_expression
        )
    if sparsity_reg > 0:
        loss = loss + metagene_entropy_loss(u, sparsity_reg)
    if metagene_reg_strength > 0:
        loss = loss + min_volume_loss(v, metagene_reg_strength)
    if gene_floor_reg > 0:
        loss = loss + gene_floor_loss(model.v, gene_floor_reg, margin=gene_floor_margin)
    if dirichlet_reg > 0:
        loss = loss + dirichlet_purity_loss(
            model.encoder.last_composition, dirichlet_reg, alpha=dirichlet_alpha
        )

    # GMM prior on u, as a "regularized information maximization" clustering
    # objective (Krause et al. 2010; Hu et al. IMSAT 2017) plus the Gaussian
    # fit. All per-cell terms are *summed* over the batch so they sit on the
    # same scale as the summed poisson reconstruction above -- with a .mean()
    # the prior was ~batch_size× too weak to shape u at all (it merely fit the
    # components to a latent space reconstruction alone had already organized).
    if mixture_strength > 0 and π is not None and model.mixture_prior == "generative":
        # Proper VaDE ELBO for the mixture prior. The reconstruction above already
        # used the sampled z (see forward), so here we add the KL between q(z,c|x)
        # and the mixture prior p(z,c). All terms summed over the batch to match
        # the summed reconstruction. z_mean and the encoder log-variance come back
        # in aux; μ_c, σ²_c are the component parameters; γ = π are the
        # responsibilities evaluated at the sampled z.
        assert aux is not None
        B = π.shape[0]
        z_mean, enc_log_var = aux  # [B,k], [k] (global)
        enc_var = torch.exp(enc_log_var)  # [k]
        var_c = torch.exp(model.component_log_vars) + 1e-6  # [K,k]
        diff = z_mean.unsqueeze(1) - model.component_means.unsqueeze(0)  # [B,K,k]
        # E_q E_γ[-log p(z|c)] (drop 2π): 0.5 Σ_c γ_c Σ_j[logσ²_c + σ²_x/σ²_c + (m-μ)²/σ²_c]
        per_ck = (
            model.component_log_vars.unsqueeze(0)
            + enc_var.view(1, 1, -1) / var_c.unsqueeze(0)
            + diff**2 / var_c.unsqueeze(0)
        )  # [B,K,k]
        gauss_kl = 0.5 * (π * per_ck.sum(dim=-1)).sum()
        # -E_q[log q(z|x)] = +0.5 Σ_j(1 + logσ²_x) per cell (encoder entropy).
        qz_entropy = -0.5 * (1.0 + enc_log_var).sum() * B
        # E_γ[log q(c|z)] responsibility entropy (drives confident assignment);
        # -E_γ[log p(c)] is constant for uniform π and dropped.
        resp_entropy = (π * torch.log(π + 1e-8)).sum()
        vade_kl = gauss_kl + qz_entropy + resp_entropy

        # Keep the marginal-entropy load-balancing term (anti-collapse); it is
        # orthogonal to the ELBO and still needed to keep all components alive.
        π_bar = π.mean(dim=0)
        marg_entropy = -(π_bar * torch.log(π_bar + 1e-8)).sum()

        gmm_loss = vade_kl - balance_weight * B * marg_entropy
        loss = loss + mixture_strength * gmm_loss

        if var_reg > 0:
            loss = loss + var_reg * (model.component_log_vars**2).sum()

    elif mixture_strength > 0 and π is not None:
        B = π.shape[0]

        # (a) Gaussian fit: pull each cell's u toward its responsible
        # component(s) and vice-versa (maximize log p(u)).
        nll = -log_mixture_prob.sum()

        # (b) Conditional entropy H(y|x): minimize -> each cell commits to one
        # component (confident, sharp responsibilities).
        cond_entropy = -(π * torch.log(π + 1e-8)).sum(dim=-1).sum()

        # (c) Marginal entropy H(y-bar): MAXIMIZE (note the minus sign) -> the
        # batch-averaged assignment stays spread across all components, which is
        # what actually prevents component collapse. Without this, (a)+(b) have a
        # degenerate optimum where every cell picks one component and the rest
        # die (exactly the collapse observed: 1 component held 45% of cells,
        # 4 were unused, and the abundant cell type replicated across several).
        # Scaled by B so this batch-level scalar (bounded by log K) is
        # commensurate with the summed per-cell terms.
        π_bar = π.mean(dim=0)
        marg_entropy = -(π_bar * torch.log(π_bar + 1e-8)).sum()

        gmm_loss = (
            nll + entropy_weight * cond_entropy - balance_weight * B * marg_entropy
        )
        loss = loss + mixture_strength * gmm_loss

        # (d) Anti-catch-all variance prior: log-normal on σ² centered at 1
        # (log σ² = 0). The Gaussian normalizer already penalizes inflating
        # variance, but per-factor diagonal variances (K×k free params) can
        # still let a component balloon into a flat catch-all that swallows
        # unrelated cells (observed: σ² up to ~12 on some components vs ~0.3 on
        # others). This keeps component scales comparable and well-conditioned.
        if var_reg > 0:
            loss = loss + var_reg * (model.component_log_vars**2).sum()

    return loss


def poisson_logprob_sparse(
    λ: torch.Tensor,
    X: torch.Tensor,
    constant_terms: bool = False,
    row_idx: torch.Tensor | None = None,
):
    """Log probability for sparse CSR input under Poisson likelihood."""
    col_idx = X.col_indices()
    if row_idx is None:
        row_idx, col_idx = _sparse_row_col_indices(X)
    x_data = X.values()
    lp = (x_data * torch.log(λ[row_idx, col_idx].clamp(1e-8))).sum() - λ.sum()
    if constant_terms:
        lp -= torch.lgamma(x_data + 1).sum()

    return lp


def nb_logprob_sparse(
    λ: torch.Tensor,
    X: torch.Tensor,
    log_r: torch.Tensor,
    constant_terms: bool = False,
    row_idx: torch.Tensor | None = None,
):
    """Log probability for sparse CSR input under a negative-binomial likelihood
    with per-gene dispersion r_g = exp(log_r), mean μ = λ (dense).

    Mirrors poisson_logprob_sparse's sparsity trick. The NB logpmf is

        lgamma(x+r) - lgamma(r) - lgamma(x+1) + r·log r - (x+r)·log(r+μ) + x·log μ

    which does not collapse to −λ at x=0 (unlike Poisson), so the zero-entries
    still carry a μ-dependent term r·log(r/(r+μ)). We compute that baseline
    densely over all entries (μ is already dense) and add the x>0 correction over
    the nonzeros only.
    """
    r = torch.exp(log_r)  # [n_genes], broadcasts over cells
    col_idx = X.col_indices()
    if row_idx is None:
        row_idx, col_idx = _sparse_row_col_indices(X)
    x_data = X.values()
    μ = λ.clamp(1e-8)
    log_r_plus_μ = torch.log(r + μ)  # [B, n_genes]

    # x=0 baseline over every cell×gene entry: r·log r − r·log(r+μ).
    baseline = (r * (torch.log(r) - log_r_plus_μ)).sum()

    # x>0 correction on the nonzeros: everything the baseline omitted.
    r_nz = r[col_idx]
    lp_nz = (
        torch.lgamma(x_data + r_nz)
        - torch.lgamma(r_nz)
        + x_data * torch.log(μ[row_idx, col_idx])
        - x_data * log_r_plus_μ[row_idx, col_idx]
    ).sum()
    if constant_terms:
        lp_nz = lp_nz - torch.lgamma(x_data + 1).sum()

    return baseline + lp_nz


class FactorizationModel:
    X: csr_matrix
    inflow: csr_matrix | None
    φ: csr_matrix | None
    device: torch.device
    model: FactorizationVAE
    m: int
    n: int

    def __init__(
        self,
        data: SpatialData | AnnData,
        n_factors: int,
        include_diffusion: bool = True,
        diffusion_aware_encoder: bool = False,
        decontaminate_encoder: bool = False,
        diffusion_input_mode: str = "subtractive",
        per_cell_alpha: bool = False,
        include_retention: bool = True,
        include_delta: bool = True,
        metagene_activation: str = "softmax",
        encoder_architecture: str = "joint",
        composition_activation: str = "softmax",
        alpha_reg: float = 0.0,
        sparsity_reg: float = 0.0,
        metagene_reg_type: str = "none",
        metagene_reg_strength: float = 0.0,
        gene_floor_reg: float = 0.0,
        gene_floor_margin: float = -3.0,
        dirichlet_reg: float = 0.0,
        dirichlet_alpha: float = 0.5,
        init_method: str = "nndsvd",
        n_components: int | None = None,
        mixture_strength: float = 1.0,
        entropy_weight: float = 0.05,
        balance_weight: float = 1.0,
        var_reg: float = 1.0,
        resp_temperature: float = 2.0,
        likelihood: str = "poisson",
        mixture_space: str = "clr",
        mixture_prior: str = "generative",
        component_init: str = "random",
    ):
        # NOTE on the latent GMM (only active when n_components is set; OFF by
        # default). The mixture was introduced to give the decontamination a
        # cell-type inductive bias, but on the honest metrics (probe separability +
        # Leiden ARI on the residual corrected expression) it made the product
        # *worse* -- it pulls u toward cluster prototypes, which makes the
        # correction over-aggressive (removes ~2x more mass) and over-fragments
        # Leiden, for no leakage benefit -- so it is no longer on by default. It
        # remains available (set n_components) and, when used, the generative
        # (VaDE) prior is the good version: mixture_strength=1.0 is the proper ELBO
        # β; balance_weight drives load-balancing; var_reg conditions the component
        # variances; entropy_weight/resp_temperature only affect the older
        # mixture_prior="regularizer" path. See examples/correction-benchmark and
        # segreg.evaluation.
        if dirichlet_reg > 0 and encoder_architecture != "composition_abundance":
            raise ValueError(
                "dirichlet_reg > 0 requires encoder_architecture="
                "'composition_abundance' (it regularizes the composition "
                "simplex, which only exists for that encoder)."
            )
        adata, self.X, self.inflow, outflow, self.φ = load_proseg_data(
            data, include_diffusion
        )

        self.m, self.n = adata.shape
        self.n_factors = n_factors
        self.var_names = adata.var_names
        self.obs_names = adata.obs_names

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.alpha_reg = alpha_reg
        self.sparsity_reg = sparsity_reg
        if metagene_reg_type not in ("none", "min_volume"):
            raise ValueError(
                f"Unknown metagene_reg_type: {metagene_reg_type}. "
                f"Must be one of: 'none', 'min_volume'."
            )
        self.metagene_reg_type = metagene_reg_type
        self.metagene_reg_strength = (
            metagene_reg_strength if metagene_reg_type == "min_volume" else 0.0
        )
        self.gene_floor_reg = gene_floor_reg
        self.gene_floor_margin = gene_floor_margin
        self.dirichlet_reg = dirichlet_reg
        self.dirichlet_alpha = dirichlet_alpha
        self.n_components = n_components
        self.mixture_strength = mixture_strength
        self.entropy_weight = entropy_weight
        self.balance_weight = balance_weight
        self.var_reg = var_reg
        self.resp_temperature = resp_temperature

        mean_expr_raw = np.asarray(self.X.mean(axis=0)).squeeze().astype(np.float32)
        log_mean_expr = torch.tensor(np.log(mean_expr_raw + 1e-4), dtype=torch.float32)

        self.gene_expression = torch.tensor(mean_expr_raw, dtype=torch.float32)

        v_init = None
        if init_method == "nndsvd":
            v_init = _nndsvd_h_init(self.X, n_factors)

        if component_init not in ("kmeans", "random"):
            raise ValueError(
                f"Unknown component_init: {component_init}. Must be 'kmeans' or 'random'."
            )
        component_means_init = None
        component_log_vars_init = None
        if (
            n_components is not None
            and component_init == "kmeans"
            and v_init is not None
        ):
            component_means_init, component_log_vars_init = _kmeans_component_init(
                self.X, v_init, n_components, mixture_space
            )

        self.model = FactorizationVAE(
            self.n,
            n_factors,
            v_init,
            include_diffusion=include_diffusion,
            diffusion_aware_encoder=diffusion_aware_encoder,
            decontaminate_encoder=decontaminate_encoder,
            diffusion_input_mode=diffusion_input_mode,
            per_cell_alpha=per_cell_alpha,
            include_retention=include_retention,
            include_delta=include_delta,
            metagene_activation=metagene_activation,
            encoder_architecture=encoder_architecture,
            composition_activation=composition_activation,
            gene_expression=self.gene_expression,
            n_components=n_components,
            resp_temperature=resp_temperature,
            likelihood=likelihood,
            mixture_space=mixture_space,
            mixture_prior=mixture_prior,
            component_means_init=component_means_init,
            component_log_vars_init=component_log_vars_init,
        )

    def fit(
        self,
        nepochs: int = 200,
        batch_size: int = 1024,
        lr: float = 1e-2,
        seed: int | None = 42,
        compile: bool = False,
        quiet: bool = False,
        grad_clip: float | None = 1.0,
        lr_schedule: str = "cosine",
        patience: int = 80,
        min_delta: float = 1e-5,
        sparsity_annealing: bool = True,
        mixture_warmup: bool = True,
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        if lr_schedule == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=nepochs, eta_min=lr * 1e-3
            )
        elif lr_schedule == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=10, min_lr=lr * 1e-3
            )
        elif lr_schedule == "none":
            scheduler = None
        else:
            raise ValueError(
                f"Unknown lr_schedule: {lr_schedule}. "
                f"Must be one of: 'cosine', 'reduce_on_plateau', 'none'."
            )

        if compile:
            print("Compiling model...")
            train_loss_fn = torch.compile(loss_fn)
        else:
            train_loss_fn = loss_fn

        best_loss = float("inf")
        no_improvement_count = 0

        # inflow/φ are unused by FactorizationVAE.forward when include_diffusion=False,
        # but SparseBatchSampler always requires same-shape matrices, so substitute zeros.
        empty = csr_matrix(self.X.shape, dtype=np.float32)
        inflow_for_sampler = self.inflow if self.inflow is not None else empty
        φ_for_sampler = self.φ if self.φ is not None else empty
        batch_sampler = SparseBatchSampler(
            self.X, inflow_for_sampler, φ_for_sampler, batch_size, device
        )

        self.model.to(self.device)
        self.model.train()

        with tqdm(range(nepochs), desc="Training", unit="epoch", disable=quiet) as pbar:
            for epoch in pbar:
                if sparsity_annealing and nepochs > 1:
                    progress = min(1.0, (epoch + 1) / (nepochs // 2))
                    current_sparsity_reg = self.sparsity_reg * progress
                else:
                    current_sparsity_reg = self.sparsity_reg

                # Warm the GMM prior in over the first ~40% of training. Applying
                # full mixture pressure to a still-random u collapses components
                # before the latent has organized into anything cluster-shaped;
                # letting reconstruction establish structure first, then ramping
                # in the prior, is markedly more robust.
                if self.n_components is not None and mixture_warmup and nepochs > 1:
                    warmup_frac = 0.4
                    current_mixture_strength = self.mixture_strength * min(
                        1.0, (epoch + 1) / max(1, int(nepochs * warmup_frac))
                    )
                else:
                    current_mixture_strength = self.mixture_strength

                batch_sampler.shuffle()
                epoch_loss_sum = 0.0
                epoch_batch_count = 0

                for (
                    X_batch,
                    inflow_batch,
                    φ_batch,
                    precomputed_row_idx,
                ) in batch_sampler:
                    optimizer.zero_grad(set_to_none=True)
                    loss = train_loss_fn(
                        self.model,
                        X_batch,
                        inflow_batch,
                        φ_batch,
                        row_idx=precomputed_row_idx,
                        alpha_reg=self.alpha_reg,
                        sparsity_reg=current_sparsity_reg,
                        metagene_reg_strength=self.metagene_reg_strength,
                        gene_floor_reg=self.gene_floor_reg,
                        gene_floor_margin=self.gene_floor_margin,
                        dirichlet_reg=self.dirichlet_reg,
                        dirichlet_alpha=self.dirichlet_alpha,
                        mixture_strength=current_mixture_strength
                        if self.n_components is not None
                        else 0.0,
                        entropy_weight=self.entropy_weight,
                        balance_weight=self.balance_weight,
                        var_reg=self.var_reg,
                    )
                    loss.backward()
                    if grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), grad_clip
                        )
                    optimizer.step()

                    epoch_loss_sum += loss.detach().item()
                    epoch_batch_count += 1
                    pass

                if not np.isfinite(epoch_loss_sum):
                    raise ValueError(f"Non-finite loss: {epoch_loss_sum}")

                # Step LR scheduler once per epoch.
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(epoch_loss_sum)
                elif scheduler is not None:
                    scheduler.step()

                if best_loss - epoch_loss_sum > min_delta:
                    best_loss = epoch_loss_sum
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                pbar.set_postfix(
                    loss=f"{epoch_loss_sum:.4f}",
                    best=f"{best_loss:.4f}",
                    patience=f"{no_improvement_count}/{patience}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2g}",
                )

                if no_improvement_count >= patience:
                    pbar.write(
                        f"Early stopping at epoch {epoch + 1}: no improvement for {patience} epochs"
                    )
                    break

    def _encode_chunk(
        self, start_idx: int, end_idx: int, α: torch.Tensor | None
    ) -> torch.Tensor:
        """Runs FactorizationVAE.encode on one row-chunk, fetching inflow/phi only if
        this model's encoder configuration actually needs them."""
        X_chunk_tensor = self._chunk_csr_tensor(self.X, start_idx, end_idx)
        inflow_chunk_tensor = None
        φ_chunk_tensor = None
        if self.model.diffusion_aware_encoder or self.model.decontaminate_encoder:
            assert self.inflow is not None and self.φ is not None
            inflow_chunk_tensor = self._chunk_csr_tensor(
                self.inflow, start_idx, end_idx
            )
            φ_chunk_tensor = self._chunk_csr_tensor(self.φ, start_idx, end_idx)
        return self.model.encode(X_chunk_tensor, inflow_chunk_tensor, φ_chunk_tensor, α)

    def get_factor_loadings(self, batch_size: int = 4096) -> np.ndarray:
        """Returns non-negative W matrix (after softplus), shape (n_cells, n_factors)."""

        Xnmf = np.zeros((self.m, self.n_factors), dtype=np.float32)
        self.model.eval()
        with torch.no_grad():
            α = torch.exp(self.model.log_α) if self.model.include_diffusion else None
            for start_idx in range(0, self.m, batch_size):
                end_idx = min(start_idx + batch_size, self.m)
                encoded_chunk = self._encode_chunk(start_idx, end_idx, α)
                Xnmf[start_idx:end_idx, :] = encoded_chunk.cpu().numpy()

        return Xnmf

    def get_factor_programs(self) -> np.ndarray:
        """Returns non-negative H matrix (after softplus), shape (n_factors, n_genes)."""
        return self.model.v_norm().detach().cpu().numpy().T

    def _chunk_csr_tensor(self, M: csr_matrix, start: int, end: int) -> torch.Tensor:
        chunk = M[start:end, :]
        return torch.sparse_csr_tensor(
            chunk.indptr,
            chunk.indices,
            chunk.data,
            size=(end - start, self.n),
            dtype=torch.float32,
            device=self.device,
        )

    def get_corrected_expression(self, batch_size: int = 4096, eps: float = 1e-8) -> csr_matrix:
        """Residual (responsibility-weighted) decontaminated expression: X * λ/μ.

        λ = u @ v_norm() is the cell's own (contamination-free) rate; μ =
        retention*λ + α*inflow is the model's reconstruction of the *observed*
        rate. Multiplying the raw counts X by λ/μ keeps the raw data's full
        per-cell resolution and its genuine zeros (the output has X's exact
        sparsity pattern), reweighting each observed count by the model's estimate
        that it is own signal rather than inflow, and dividing by retention to
        restore signal lost to outflow.

        This replaces the earlier product (the bare low-rank rate λ). λ manufactures
        cross-type signal -- softmax over factors gives every cell a nonzero dose of
        every metagene, smearing markers across cell types -- and over-smooths, so
        it scored WORST on the honest metrics (probe separability, Leiden ARI, and a
        scale-invariant marker-leakage ratio: λ's leakage was ~3x raw). The residual
        Pareto-dominates it. See examples/correction-benchmark/eval-harness.py /
        segreg.evaluation. Without diffusion, μ=λ and this reduces to X (no
        contamination model -> nothing to correct). The low-rank λ is still
        available as get_factor_loadings() @ get_factor_programs()."""
        self.model.eval()

        rows, cols, data = [], [], []
        with torch.no_grad():
            vnorm = self.model.v_norm()
            α_enc = torch.exp(self.model.log_α) if self.model.include_diffusion else None
            for start_idx in range(0, self.m, batch_size):
                end_idx = min(start_idx + batch_size, self.m)

                u_chunk = self._encode_chunk(start_idx, end_idx, α_enc)
                λ = u_chunk @ vnorm  # [c, n] own rate
                X = self._chunk_csr_tensor(self.X, start_idx, end_idx).to_dense()
                if self.model.include_diffusion:
                    assert self.inflow is not None and self.φ is not None
                    inflow_s = self._chunk_csr_tensor(self.inflow, start_idx, end_idx)
                    φ_s = self._chunk_csr_tensor(self.φ, start_idx, end_idx)
                    α = self.model.alpha(inflow_s, φ_s)  # [n] or [c, n]
                    retention = torch.exp(-α * φ_s.to_dense())
                    μ = retention * λ + α * inflow_s.to_dense()
                else:
                    μ = λ
                # X's zeros stay zero; nonzeros are reweighted by own-signal fraction.
                corrected = (X * λ / μ.clamp(min=eps)).cpu().numpy()

                r, c = np.nonzero(corrected)
                rows.append(r + start_idx)
                cols.append(c)
                data.append(corrected[r, c])

        return csr_matrix(
            (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
            shape=(self.m, self.n),
        )

    def get_latent_representation(self, batch_size: int = 4096) -> np.ndarray:
        """Returns the non-negative cell factor loadings W (after softplus)."""
        return self.get_factor_loadings()

    def get_cluster_assignments(
        self, batch_size: int = 4096
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns cluster assignments from the GMM prior on U.

        Returns:
            assignments: Array of shape (n_cells,) with cluster indices (argmax of π)
            π: Array of shape (n_cells, n_components) with mixture weights
        """
        if self.n_components is None:
            raise ValueError("Model was not initialized with n_components")

        self.model.eval()
        π_list = []

        with torch.no_grad():
            α = torch.exp(self.model.log_α) if self.model.include_diffusion else None
            for start_idx in range(0, self.m, batch_size):
                end_idx = min(start_idx + batch_size, self.m)
                u_chunk = self._encode_chunk(start_idx, end_idx, α)
                _, γ = self.model.compute_gmm_loss(u_chunk)
                assert γ is not None
                π_list.append(γ.cpu().numpy())

        π_all = np.concatenate(π_list, axis=0)
        assignments = np.argmax(π_all, axis=-1)

        return assignments, π_all
