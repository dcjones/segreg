
from typing import cast

from anndata import AnnData
from patsy import dmatrix
from patsy.design_info import DesignMatrix
from scipy.sparse import coo_matrix, csr_matrix
from spatialdata import SpatialData
from torch import Tensor
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn

from .loader import RegressionBatch, RegressionBatchLoader


class Regression:
    # count matrix point estimate
    X: csr_matrix

    # sparse mixing matrix between cell pairs. Rows sum to 1.
    A: csr_matrix
    bg_mix_rate: npt.NDArray[np.float32]

    design: DesignMatrix

    # [ncells, 2] cell centroids, used to form spatially local batches
    spatial: npt.NDArray[np.floating] | None

    def __init__(self, data: SpatialData | AnnData, formula: str):
        if isinstance(data, SpatialData):
            adata = data.tables["table"]
        elif isinstance(data, AnnData):
            adata = data
        else:
            raise TypeError("data must be an AnnData or SpatialData object")

        # In anndata X can be practically anything array like, but proseg always outputs csr matrices
        assert isinstance(adata.X, csr_matrix)
        self.X = adata.X

        # Construct design matrix
        design_df = dmatrix(formula, adata.obs, return_type="dataframe")
        self.design = cast(DesignMatrix, design_df)

        # Construct the neighbor mixing matrix
        state_transitions = adata.obsp["state_transitions"]
        assert isinstance(state_transitions, csr_matrix)

        # TODO: Okay, where does from_bg_trans_count come into play? Do we actually need that?
        # assert "from_bg_trans_count" in adata.obs
        # from_bg_trans_count = np.asarray(adata.obs["from_bg_trans_count"])

        assert "to_bg_trans_count" in adata.obs
        to_bg_trans_count = np.asarray(adata.obs["to_bg_trans_count"])

        total_transitions = np.asarray(state_transitions.sum(axis=1)).squeeze() + to_bg_trans_count
        A = state_transitions / total_transitions[:, None]
        assert isinstance(A, coo_matrix)
        A = A.tocsr()
        assert isinstance(A, csr_matrix)
        self.A = A.astype(np.float32)

        self.bg_mix_rate = (to_bg_trans_count / total_transitions).astype(np.float32)

        self.spatial = np.asarray(adata.obsm["spatial"]) if "spatial" in adata.obsm else None

    def fit(
        self,
        nepochs: int = 1,
        batch_size: int = 1024,
        seed: int | None = None,
        device: torch.device | str | None = None,
    ):
        loader = RegressionBatchLoader(
            self.X,
            self.A,
            self.bg_mix_rate,
            np.asarray(self.design, dtype=np.float32),
            batch_size,
            spatial=self.spatial,
            seed=seed,
            device=device,
        )

        for _epoch in range(nepochs):
            for batch in loader:
                pass

                # TODO: training

    def regression_coefficients(self):
        pass


class RegressionModel(nn.Module):
    """
    Basic regression with inter-cell mixing/missegmentation according to some fixed mixing matrix.
    """

    include_mixing: bool

    def __init__(self, ncells: int, ngenes: int, ncovariates: int, include_mixing: bool = True):
        super().__init__()

        self.include_mixing = include_mixing

        # per-gene regression intercept
        self.gene_bias = nn.Parameter(torch.zeros(ngenes))

        # background expression rates
        self.bg_rates = nn.Parameter(torch.zeros(ngenes))

        # negative-binomial overdispersion parameters
        self.log_r = nn.Parameter(torch.full((ngenes,), 2.0))

        # regression coefficient surrogate posterior parameters
        self.β_μ = nn.Parameter(torch.full((ncovariates, ngenes), 0.0))
        self.β_logσ = nn.Parameter(torch.full((ncovariates, ngenes), -2.0))


    def forward(self, batch: RegressionBatch):
        if self.training:
            β_σ = torch.exp(self.β_logσ.clamp(max=4.0))
            β = self.β_μ + torch.randn_like(β_σ) * β_σ
        else:
            β = self.β_μ

        # regression
        # TODO: if this proves to be too inflexible, we may have to revive our old scheme of encoding
        # some amount of extra per-cell variation using a VAE term.
        λ = torch.exp(batch.design @ β + self.gene_bias)

        λ_obs = self.mix(λ, batch)

        # Senders-only nodes exist to supply λ for the mixing; only receivers are
        # modeled, so the likelihood is over the leading block of the batch.
        X_obs = batch.X[: batch.nreceivers, :]

        # TODO: compute NB log-likelihood
        ll = self.negbinom_likelihood(λ_obs, X_obs)

        # TODO: compute KL term (for beta)
        return -ll

    def mix(self, λ: Tensor, batch: RegressionBatch) -> Tensor:
        """Corrupt per-cell rates by missegmentation, [nnodes, ngenes] -> [nreceivers, ngenes].

        A[i,j] is the posterior probability that a molecule counted in cell i was
        actually assigned to cell j, and bg_mix_rate[i] the probability it was
        background, so A[i,:].sum() + bg_mix_rate[i] == 1 and λ_obs is a convex
        combination of the neighborhood's true rates. The self term is already in
        there: A has a dominant diagonal (~60% of each row).
        """
        nr = batch.nreceivers
        if not self.include_mixing:
            return λ[:nr, :]

        # [nedges, ngenes], one row per (receiver, sender) pair
        contrib = batch.weights.unsqueeze(1) * λ[batch.senders, :]

        λ_obs = torch.zeros((nr, λ.shape[1]), dtype=λ.dtype, device=λ.device)
        λ_obs.index_add_(0, batch.receivers, contrib)

        # background is the remaining mixture component
        return λ_obs + batch.bg_weight[:nr].unsqueeze(1) * torch.exp(self.bg_rates)

    def negbinom_likelihood(self, λ: Tensor, X: Tensor) -> Tensor:
        r = torch.exp(self.log_r).clamp(min=1e-3)
        eps = 1e-8
        log_r_over_r_plus_mu = torch.log(r / (r + λ + eps))
        log_mu_over_r_plus_mu = torch.log((λ + eps) / (r + λ + eps))
        return (
                torch.lgamma(X + r)
                - torch.lgamma(r)
                - torch.lgamma(X + 1)
                + r * log_r_over_r_plus_mu
                + X * log_mu_over_r_plus_mu
            ).sum(dim=-1)
