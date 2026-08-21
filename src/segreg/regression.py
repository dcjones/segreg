
import warnings
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
import pandas as pd
import scipy.stats as stats

from .loader import RegressionBatch, RegressionBatchLoader


class Regression:
    # count matrix point estimate
    X: csr_matrix

    # sparse mixing matrix between cell pairs. Rows sum to 1.
    A: csr_matrix
    bg_mix_rate: npt.NDArray[np.float32]

    design: DesignMatrix

    # [ncells] log total counts, used as a fixed regression offset
    log_size: npt.NDArray[np.float32]

    # [ncells, 2] cell centroids, used to form spatially local batches
    spatial: npt.NDArray[np.floating] | None

    # set by fit()
    model: "RegressionModel | None" = None

    # gene names copied from the anndata
    var_names: list[str]

    # true if the estimated mixing matrix should be used
    include_mixing: bool

    def __init__(self, data: SpatialData | AnnData, formula: str, include_mixing: bool=True,
                 mixing_interval_correction: bool=False, include_drift: bool=False,
                 drift: pd.DataFrame | None=None, include_cellspace: bool=False,
                 cellspace: dict | None=None, cellspace_percluster: bool=False,
                 cellspace_shrink_σ: float=1.0):
        if isinstance(data, SpatialData):
            adata = data.tables["table"]
        elif isinstance(data, AnnData):
            adata = data
        else:
            raise TypeError("data must be an AnnData or SpatialData object")

        self.var_names = list(adata.var_names)
        self.include_mixing = include_mixing
        # Widen the reported intervals for the correlation that molecule
        # reallocation induces between cells sharing a donor. Off by default, and
        # it changes only posterior_sd() -- never the fit, never a point estimate.
        self.mixing_interval_correction = mixing_interval_correction
        self.include_drift = include_drift
        # --- the CELL-SPACE alternative to the drift term ------------------------
        # `include_drift` adds kappa*u to the COEFFICIENT, where u is frac projected
        # onto the design; this adds kappa*frac[i,g] to each cell's linear predictor
        # directly, keeping the component of frac ORTHOGONAL to the design that the
        # projection discards.
        #
        # Why it might win: simple-seg-sim DESIGN_v2 §60 measured that the rank-one
        # residual is reproducibly structured (cross-seed rho 0.31-0.34) but is
        # explained by NO analytic second direction we could construct (best
        # |corr| 0.12 against headroom 0.19-0.24 R²). Using the full field requires
        # no such guess.
        #
        # Why it might lose: (d/s - 1) is wildly heterogeneous (IQR 1.3 breast,
        # 2.8 Atera), so a single kappa is an average either way -- and the
        # coefficient-space version averages it over exactly the direction that
        # biases beta, while this averages over the whole field, most of which is
        # orthogonal to the design and cannot bias anything.
        #
        # frac is never materialized (3.0 GB on Atera); the pieces are held and
        # recombined per minibatch. See cellspace_frac().
        self.include_cellspace = include_cellspace
        # RUNG 2 of the scale ladder. With one global kappa the cell-space term
        # INJECTS bias: fitted at -2.31 (breast) / -3.43 (Atera) it lands on an
        # average dominated by the large tumor types and then distorts every type
        # whose frac-design alignment differs from that average -- breast Invasive
        # Tumor's null bias went +0.005 -> +0.370 and its FP 0.273 -> 0.870
        # (simple-seg-sim DESIGN_v2 §61).
        #
        # That is also what made the §61 comparison UNFAIR: the drift arms carry
        # kappa per design column, which is kappa per cell type (13 breast, 9
        # Atera), against a single scalar here. Per-cluster kappa is the like-for-
        # like test -- Leiden clusters are annotation-free and at least as granular
        # (56 breast, 225 Atera).
        #
        # Parameterized as kappa_c = kappa0 + delta_c with delta penalized, so the
        # cluster count does not buy unchecked freedom against a likelihood that is
        # exactly flat in this direction per parameter.
        self.cellspace_percluster = cellspace_percluster
        self.cellspace_shrink_σ = cellspace_shrink_σ
        self.cellspace = None
        if include_cellspace:
            if cellspace is None:
                raise ValueError(
                    "include_cellspace=True requires `cellspace`: the dict written "
                    "by bench/scripts/cluster_mixing_covariates.py --reduce field "
                    "(M, sw, own, P, original_cell_id, genes).")
            self.cellspace = cellspace
        elif cellspace is not None:
            raise ValueError("`cellspace` supplied but include_cellspace is False")

        # In anndata X can be practically anything array like, but proseg always outputs csr matrices
        assert isinstance(adata.X, csr_matrix)
        self.X = adata.X

        # Construct design matrix
        # TODO: consider checking if the design has an intercept and
        # excluding the redundant gene_bias parameter in RegressionModel if so.
        design_df = dmatrix(formula, adata.obs, return_type="dataframe")
        self.design = cast(DesignMatrix, design_df)

        # --- the DRIFT covariate ------------------------------------------------
        # u[k, g] is the exposure slope of gene g's foreign share along design
        # column k -- the direction along which missegmentation biases a
        # coefficient. Supplied rather than computed here: it depends on an
        # unsupervised clustering of the counts, which is a preprocessing choice
        # and does not belong inside the model.
        #
        # WHY THIS IDENTIFIES. The mean model fits design @ (β + κ*u), so β and δ
        # = κ*u compete to explain the same exposure-correlated variation. What
        # separates them is that β carries the N(0, β_prior_σ²) KL prior and κ does
        # not, so variation that u can explain is explained by κ -- one scalar per
        # design column -- leaving β shrunk toward zero. That asymmetry, not an
        # extra prior on κ, is the mechanism: a coefficient is pulled to zero
        # exactly where the data are consistent with contamination. Which is also
        # why κ has no prior of its own; adding one would fight the mechanism.
        #
        # Reported coefficients are β, NOT β + δ, so get_regression_coefficients()
        # returns the de-biased estimate without any change.
        self.drift = None
        if include_drift:
            if drift is None:
                raise ValueError(
                    "include_drift=True requires `drift`: a DataFrame of the drift "
                    "covariate, indexed by DESIGN COLUMN NAME with one column per "
                    "gene. Rows/columns not present are filled with 0, so a "
                    "covariate that only covers the exposure columns is fine."
                )
            cols = list(self.design.design_info.column_names)
            aligned = drift.reindex(index=cols, columns=self.var_names)
            # A silent misalignment produces an all-zero u and an arm that looks
            # like a no-op rather than like a bug, so it is an error.
            n_ok = int(aligned.notna().any(axis=1).sum())
            if n_ok == 0:
                raise ValueError(
                    "`drift` shares no index value with the design's column names. "
                    f"Design columns look like {cols[:3]}; drift index looks like "
                    f"{list(drift.index[:3])}. The index must be design column "
                    "names, not cell-type labels."
                )
            self.drift = aligned.fillna(0.0).to_numpy(dtype=np.float32)
        elif drift is not None:
            raise ValueError("`drift` was supplied but include_drift is False")

        # Construct the neighbor mixing matrix
        state_transitions = adata.obsp["state_transitions"]
        assert isinstance(state_transitions, csr_matrix)

        # TODO: Okay, where does from_bg_trans_count come into play? Do we actually need that?
        # assert "from_bg_trans_count" in adata.obs
        # from_bg_trans_count = np.asarray(adata.obs["from_bg_trans_count"])

        assert "to_bg_trans_count" in adata.obs
        to_bg_trans_count = np.asarray(adata.obs["to_bg_trans_count"])

        total_transitions = np.asarray(state_transitions.sum(axis=1)).squeeze() + to_bg_trans_count

        # A cell with no transcripts has no recorded transitions either, so this
        # normalization would be 0/0. Dividing by 1 instead leaves such a cell
        # with an all-zero mixing row and a zero background weight, which makes
        # its λ_obs exactly 0 against an all-zero X: a likelihood term of ~0 that
        # contributes to no gradient. That is equivalent to dropping the cell,
        # but keeps the batch indexing and the cell ordering intact. Without the
        # guard the NaN reaches every shared parameter on the first step and the
        # whole fit is NaN.
        empty = total_transitions == 0
        if empty.any():
            warnings.warn(
                f"{int(empty.sum())} of {len(empty)} cells have no recorded state "
                "transitions (usually cells with no transcripts). They will not "
                "contribute to the fit.",
                stacklevel=2,
            )
        normalizer = np.where(empty, 1.0, total_transitions)

        A = state_transitions / normalizer[:, None]
        assert isinstance(A, coo_matrix)
        A = A.tocsr()
        assert isinstance(A, csr_matrix)
        self.A = A.astype(np.float32)

        self.bg_mix_rate = (to_bg_trans_count / normalizer).astype(np.float32)

        # Observed totals are very nearly a fixed point of the mixing model
        # (median relative error of A @ s + bg vs s is -0.2%), so we can use them
        # as a fixed offset rather than fitting per-cell size factors.
        counts = np.asarray(self.X.sum(axis=1)).squeeze()
        self.log_size = np.log(np.maximum(counts, 1)).astype(np.float32)

        if self.cellspace is not None:
            cs = self.cellspace
            # Align to THIS object's cell order by original_cell_id and to its gene
            # order by name. A silent misalignment here would look like a weak
            # result rather than a bug, so both are errors.
            oid = np.asarray(adata.obs["original_cell_id"]).astype(np.int64)
            pos = pd.Series(np.arange(len(cs["original_cell_id"])),
                            index=np.asarray(cs["original_cell_id"]).astype(np.int64))
            pos = pos[~pos.index.duplicated()]
            row = pos.reindex(oid)
            if row.isna().any():
                raise ValueError(
                    f"`cellspace` is missing {int(row.isna().sum())} of "
                    f"{len(oid)} cells; it must cover every cell of this object.")
            gi = pd.Series(np.arange(len(cs["genes"])),
                           index=np.asarray(cs["genes"]).astype(str))
            gi = gi[~gi.index.duplicated()].reindex(
                [str(v) for v in self.var_names])
            if gi.isna().all():
                raise ValueError(
                    "`cellspace` shares no gene name with this object.")
            self._cs_gene = gi.fillna(-1).to_numpy().astype(np.int64)
            r_ = row.to_numpy().astype(int)
            self.cellspace = dict(
                M=np.ascontiguousarray(cs["M"][r_]),
                sw=np.ascontiguousarray(cs["sw"][r_]),
                own=np.ascontiguousarray(cs["own"][r_]).astype(np.int64),
                P=np.ascontiguousarray(cs["P"]),
                gene_index=self._cs_gene,
            )
            n_missing = int((self._cs_gene < 0).sum())
            if n_missing:
                warnings.warn(
                    f"{n_missing} of {len(self._cs_gene)} genes absent from "
                    "`cellspace`; their cell-space correction is 0.",
                    stacklevel=2)

        self.spatial = np.asarray(adata.obsm["spatial"]) if "spatial" in adata.obsm else None

    def fit(
        self,
        nepochs: int = 100,
        batch_size: int = 1024,
        lr: float = 0.01,
        β_prior: str = "cauchy",
        β_prior_σ: float = 1.0,
        κ_prior_μ: float = 0.0,
        κ_prior_σ: float | None = None,
        seed: int | None = None,
        device: torch.device | str | None = None,
        verbose: bool = True,
        compile: bool = True,
    ) -> "RegressionModel":
        loader = RegressionBatchLoader(
            self.X,
            self.A,
            self.bg_mix_rate,
            np.asarray(self.design, dtype=np.float32),
            self.log_size,
            batch_size,
            spatial=self.spatial,
            seed=seed,
            device=device,
        )

        # Parameter initialization is otherwise unseeded, which on its own is
        # enough to move null coefficients run to run.
        if seed is not None:
            torch.manual_seed(seed)

        # Recorded on the object, not only passed on, because it is the one
        # fit-time argument a caller may need to ASSERT is live: bench's
        # `requires:` check resolves attributes here, and "cauchy but silently
        # normal" and "cauchy" look identical in a topline.
        self.β_prior = β_prior

        ncells, ngenes = self.X.shape
        model = RegressionModel(
            ncells,
            ngenes,
            self.design.shape[1],
            include_mixing=self.include_mixing,
            β_prior=β_prior,
            β_prior_σ=β_prior_σ,
            drift=self.drift,
            κ_prior_μ=κ_prior_μ,
            κ_prior_σ=κ_prior_σ,
            cellspace=self.cellspace,
            cellspace_percluster=self.cellspace_percluster,
            cellspace_shrink_σ=self.cellspace_shrink_σ,
        )
        if device is not None:
            model = model.to(device)

        param_device = next(model.parameters()).device

        # The parameters are small enough that a step's worth of unfused Adam
        # launches costs more than the rest of the batch put together.
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, fused=param_device.type == "cuda"
        )

        # The likelihood is a long chain of elementwise ops over [ncells, ngenes]
        # blocks, so a step is bound by kernel launches rather than by the GPU.
        # Fusing them is worth roughly a factor of two. Batch shapes differ from
        # tile to tile, hence dynamic.
        #
        # Note that this changes fitted coefficients slightly even at a fixed
        # seed: inductor draws the β reparameterization noise from its own RNG
        # stream, so a compiled fit follows a different (equally valid) sample
        # path. With the noise held fixed the two agree to ~1e-7 relative.
        step = model.forward
        if compile:
            step = torch.compile(model.forward, dynamic=True)

        model.train()
        for epoch in range(nepochs):
            # Accumulated on device; reading it every batch would force a sync.
            total = torch.zeros((), device=param_device)
            for batch in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = step(batch)
                loss.backward()
                optimizer.step()
                total += loss.detach()

            if verbose:
                # Per cell, so it stays comparable across batch sizes.
                print(f"epoch {epoch + 1}/{nepochs}: loss = {float(total) / ncells:.4f}")

        model.eval()
        self.model = model
        return model

    def posterior_sd(self, chunk: int = 16, sandwich: bool | None = None
                     ) -> npt.NDArray[np.float64]:
        """Marginal posterior sd of β, [ncovariates, ngenes].

        NOT `exp(β_logσ)`. The surrogate posterior is factorized over every
        (covariate, gene) entry, so it can only represent the CONDITIONAL sd —
        every other coefficient pinned at its mean — while a credible interval for
        one coefficient needs the MARGINAL one, integrating over the others. The two
        differ by however correlated the design is, and here that is a lot: a
        per-type exposure column is supported only on that type's cells, so it
        shares its support with that type's indicator. Measured on the breast
        benchmark the marginal is up to 3.6x wider, and the gap tracks the
        false-positive rate almost exactly.

        So the width is recomputed here, from the Laplace approximation at the
        fitted β. The negative-binomial likelihood factorizes over genes and β[:,g]
        reaches only gene g, so the Hessian is block diagonal with one
        [ncov, ncov] block per gene, and the marginal sd is just
        sqrt(diag(H⁻¹)) — cheap, because the block is the width of the DESIGN, not
        the gene count.

        For a log link, d(-loglik)/dη² under Fisher information is r/(μ(r+μ)) per
        cell with η = log μ, so H = J' diag(w) J where J = dμ/dβ. Mixing enters only
        through J: λ is the cell's own rate but μ is A λ + background, so
        J = A diag(λ) X rather than diag(λ) X.

        The prior's curvature (1/β_prior_σ² per coefficient) is added, and it is not
        a formality. Under mixing, spatially-intermixed cell types lose most of their
        identifiability — Stromal's conditional sd goes 0.145 to 9.5 — and without
        the prior term those pairs would be handed enormous likelihood-only
        intervals, when the honest posterior there is prior-dominated and the fitted
        β̂ is shrunk to match.
        """
        if self.model is None:
            raise Exception("fit() must be called before posterior_sd")
        if sandwich is None:
            sandwich = getattr(self, "mixing_interval_correction", False)

        m = self.model
        dev = next(m.parameters()).device
        B = m.β_μ.detach().double()
        gene_bias = m.gene_bias.detach().double()
        r = torch.exp(m.log_r.detach().double())
        bg_profile = torch.softmax(m.bg_rates.detach().double(), dim=-1)

        X = torch.as_tensor(np.asarray(self.design, dtype=np.float64), device=dev)
        log_size = torch.as_tensor(self.log_size.astype(np.float64), device=dev)
        A = torch.sparse_csr_tensor(
            torch.as_tensor(self.A.indptr, dtype=torch.int64, device=dev),
            torch.as_tensor(self.A.indices, dtype=torch.int64, device=dev),
            torch.as_tensor(self.A.data.astype(np.float64), device=dev),
            size=self.A.shape,
        )
        bg_weight = (
            torch.as_tensor(self.bg_mix_rate.astype(np.float64), device=dev)
            * torch.exp(log_size)
        )
        if sandwich:
            # A' as a second operator. The meat below needs the mass flowing OUT of
            # each donor (a column of A); the likelihood only ever needs what flows
            # in (a row).
            at = self.A.tocoo()
            At = torch.sparse_coo_tensor(
                torch.as_tensor(np.vstack([at.col, at.row]), dtype=torch.int64,
                                device=dev),
                torch.as_tensor(at.data.astype(np.float64), device=dev),
                size=self.A.shape,
            ).coalesce()
            # A elementwise-squared, which turns the i == k term of A D A' into a
            # single sparse product rather than a dense diagonal.
            A2 = torch.sparse_coo_tensor(
                torch.as_tensor(np.vstack([at.row, at.col]), dtype=torch.int64,
                                device=dev),
                torch.as_tensor((at.data.astype(np.float64)) ** 2, device=dev),
                size=self.A.shape,
            ).coalesce()

        ncov, ngenes = B.shape
        ncells = X.shape[0]
        # The prior's contribution to the curvature. Gaussian: 1/σ², the same for
        # every coefficient. Cauchy: the local quadratic that MATCHES THE PRIOR'S
        # GRADIENT at β̂, i.e. 2β/(γ² + β²) = β/σ_eff² with precision
        # 2/(γ² + β̂²). Not the exact second derivative, 2(γ² − β̂²)/(γ² + β̂²)²,
        # which goes NEGATIVE for |β̂| > γ and would hand a shrunk-to-nothing
        # coefficient an indefinite Hessian; the gradient-matched form is positive
        # everywhere, agrees with the true curvature at β̂ = 0 (both 2/γ²), and is
        # the majorizing quadratic the scale-mixture view of the Cauchy gives. It
        # is per-(covariate, gene) rather than a scalar, which is the whole point:
        # a large coefficient is barely penalized and gets a likelihood-driven
        # interval, while a null one is prior-dominated.
        if getattr(m, "β_prior", "normal") == "cauchy":
            prior_prec = 2.0 / (float(m.β_prior_σ) ** 2 + B**2)
        else:
            prior_prec = torch.full_like(
                B, 1.0 / float(m.β_prior_σ) ** 2)
        eye = torch.eye(ncov, dtype=torch.float64, device=dev)
        out = np.empty((ncov, ngenes), dtype=np.float64)

        for start in range(0, ngenes, chunk):
            sl = slice(start, min(start + chunk, ngenes))
            G = sl.stop - sl.start
            η = X @ B[:, sl] + gene_bias[sl].unsqueeze(0) + log_size.unsqueeze(1)
            λ = torch.exp(η.clamp(max=30.0))
            if self.include_mixing:
                # [ncells, G, ncov], flattened so the sparse product is one call.
                d = (λ.unsqueeze(2) * X.unsqueeze(1)).reshape(ncells, G * ncov)
                J = (A @ d).reshape(ncells, G, ncov)
                μ = (A @ λ) + bg_weight.unsqueeze(1) * bg_profile[sl].unsqueeze(0)
            else:
                J = λ.unsqueeze(2) * X.unsqueeze(1)
                μ = λ
            w = r[sl].unsqueeze(0) / (μ * (r[sl].unsqueeze(0) + μ) + 1e-12)
            H = torch.einsum("ngc,ngd->gcd", J, J * w.unsqueeze(2))
            # [G, ncov, ncov] diagonal, since the precision now varies by
            # coefficient as well as by gene.
            H = H + prior_prec[:, sl].T.unsqueeze(2) * eye
            cov = torch.linalg.inv(H)
            if sandwich:
                # Missegmentation reallocates MOLECULES, so two cells fed by the
                # same donor share that donor's one realized expression and their
                # residuals are correlated. With a_ij the mixing weights and N_j a
                # donor's realized count, X_i = sum_j Binomial(N_j, a_ij) gives
                #
                #   Var(X_i)      = mu_i + sum_j a_ij^2 lambda_j^2 / r
                #   Cov(X_i, X_k) =        sum_j a_ij a_kj lambda_j^2 / r
                #
                # The likelihood assumes those cells are independent, so the
                # posterior concentrates as though there were more independent
                # observations than there are. The fitted r absorbs the MARGINAL
                # variance; it cannot absorb the correlation, and since exposure is
                # spatially structured the correlated part projects onto it -- which
                # is the measured failure: coverage 0.575 at nominal 0.95.
                #
                # H is already the bread of a sandwich, and the naive interval is
                # the case Cov = diag(V), for which the meat collapses back to H.
                # Substituting the covariance above gives J'W Cov WJ. The full
                # [ncells, ncells] covariance is never formed: only A'(WJ) is
                # needed, one sparse product of the shape the Jacobian already uses.
                # The donor-level dispersion is CALIBRATED, not taken from r_g.
                #
                # Two earlier attempts failed for related reasons. Substituting the
                # whole reallocation covariance with D = lambda^2/r_g made intervals
                # narrower (FP 0.182 -> 0.436), because r_g is fitted at the
                # OBSERVED level and has already absorbed the marginal
                # overdispersion, while sum_j a_ij^2 lambda_j^2 <= mu_i^2 makes the
                # reallocation-implied marginal smaller. Deleting the diagonal to
                # avoid that double-count then made the added term INDEFINITE -- an
                # off-diagonal-only matrix has zero trace -- so it widened some
                # coefficients and narrowed others (planted coverage up, null FP up).
                #
                # Both are fixed by asking what donor-level variance is CONSISTENT
                # with the dispersion the model actually fitted. Writing the donor
                # variance as D_j = c_g lambda_j^2, the implied marginal is
                # sum_j a_ij^2 D_j, and the fitted marginal overdispersion is
                # mu_i^2 / r_g; matching them in aggregate per gene gives
                #
                #     c_g = sum_i (mu_i^2 / r_g)  /  sum_i sum_j a_ij^2 lambda_j^2
                #
                # so the marginal is preserved BY CONSTRUCTION and the covariance
                # diag(mu) + A diag(D) A' stays PSD. What is left over is purely the
                # between-cell correlation the likelihood omits -- which is the term
                # this correction exists to add.
                M = J * w.unsqueeze(2)                        # [ncells, G, ncov]
                lam2 = λ * λ
                s_i = A2 @ lam2                               # sum_j a_ij^2 lam_j^2
                c_g = ((μ * μ) / r[sl].unsqueeze(0)).sum(0) / s_i.sum(0).clamp(min=1e-12)
                dvar = c_g.unsqueeze(0) * lam2                # donor-rate variance
                U = (At @ M.reshape(ncells, G * ncov)).reshape(ncells, G, ncov)
                meat = torch.einsum("ngc,ngd->gcd", M, M * μ.unsqueeze(2))
                meat = meat + torch.einsum("ngc,ngd->gcd", U, U * dvar.unsqueeze(2))
                cov = cov @ meat @ cov
            sd = torch.diagonal(cov, dim1=1, dim2=2).clamp(min=0.0).sqrt()
            out[:, sl] = sd.T.cpu().numpy()
        return out

    def get_cellspace_scales(self) -> np.ndarray:
        """The fitted cell-space kappa: a scalar, or one per Leiden cluster."""
        if self.model is None or not self.include_cellspace:
            raise Exception("no cell-space term on this fit")
        k0 = float(self.model.κ_cs.detach().cpu().numpy())
        if not getattr(self.model, "cellspace_percluster", False):
            return np.array([k0])
        return k0 + self.model.δ_cs.detach().cpu().numpy()

    def get_drift_scales(self) -> pd.Series:
        """The fitted κ, one per design column.

        Auditability, not a result: κ is the amount of each column's
        exposure-correlated signal the model attributed to missegmentation rather
        than to regulation. A κ whose sign disagrees with the drift covariate's
        construction is the sparsity assumption failing for that column, and the
        coefficients there should be read as uncorrected.
        """
        if self.model is None:
            raise Exception("fit() must be called before get_drift_scales")
        if not self.include_drift:
            raise Exception("this model was fit without include_drift")
        return pd.Series(
            self.model.κ.detach().cpu().numpy(),
            index=list(self.design.design_info.column_names), name="kappa",
        )

    def get_regression_coefficients(self, credible_interval: float | None = None) -> pd.DataFrame:
        if self.model is None:
            raise Exception("fit() must be called before get_regression_coefficients")

        β_μ = self.model.β_μ.detach().cpu().numpy()

        covariate_names = self.design.design_info.column_names
        df = (
            pd.DataFrame(β_μ, index=covariate_names, columns=self.var_names)
            .melt(ignore_index=False, var_name="Gene", value_name="Mean")
            .reset_index(names="Covariate")
            )

        if credible_interval is not None:
            β_σ = self.posterior_sd()

            z = stats.norm.ppf(1.0 - (1.0 - credible_interval) / 2.0)
            df["Lower"] = (β_μ - z * β_σ).flatten(order="F")
            df["Upper"] = (β_μ + z * β_σ).flatten(order="F")
            df["MinimumCredible"] = np.where(
                df["Lower"] > 0,
                df["Lower"],
                np.where(df["Upper"] < 0, df["Upper"], 0.0),
            )

        return df


class RegressionModel(nn.Module):
    """
    Basic regression with inter-cell mixing/missegmentation according to some fixed mixing matrix.
    """

    include_mixing: bool
    include_drift: bool

    def __init__(
        self,
        ncells: int,
        ngenes: int,
        ncovariates: int,
        include_mixing: bool = True,
        β_prior: str = "cauchy",
        β_prior_σ: float = 1.0,
        drift: np.ndarray | None = None,
        κ_prior_μ: float = 0.0,
        κ_prior_σ: float | None = None,
        cellspace: dict | None = None,
        cellspace_percluster: bool = False,
        cellspace_shrink_σ: float = 1.0,
    ):
        super().__init__()

        self.include_mixing = include_mixing
        self.include_drift = drift is not None
        self.include_cellspace = cellspace is not None
        self.cellspace_percluster = False

        # Needed to weight the KL against a minibatch's share of the likelihood.
        self.ncells = ncells

        # Prior on the regression coefficients: Cauchy(0, β_prior_σ) by DEFAULT,
        # or N(0, β_prior_σ²) with `β_prior="normal"` -- one scale parameter either
        # way, so a caller switches the FAMILY without also retuning a knob. See
        # β_kl for the two cross-entropy terms and why the Cauchy one is a
        # single-sample estimate.
        #
        # WHY THE CAUCHY IS THE DEFAULT (simple-seg-sim bench/DESIGN_v2 §63). At
        # γ=1 it is at or above the Gaussian on false positives AND on power for
        # the largest effects at the same time -- Atera cur FP 0.176 vs 0.179, FP
        # panel 0.150 vs 0.160, power at |log-FC| > 0.65 0.85 vs 0.83 -- so it is
        # the one setting on this axis that costs nothing. Tightening γ below 1
        # then buys FP at a steady ~8 points of large-effect power per 0.04 of cur
        # FP, with no sweet spot anywhere in between: γ is a DIAL for the caller,
        # not a tuned constant.
        #
        # It matters ONLY where the likelihood is flat, which in practice means the
        # drift term's β/κ split; with β facing the likelihood alone the family is
        # worth nothing (σ 1.0 vs 0.3 moves the null bias in the 4th decimal). So
        # this default is not a general claim about regularizing β.
        if β_prior not in ("normal", "cauchy"):
            raise ValueError(
                f"β_prior must be 'normal' or 'cauchy', got {β_prior!r}")
        self.β_prior = β_prior
        self.β_prior_σ = β_prior_σ

        # Per-gene regression intercept. With log_size as an offset this is a log
        # share of the cell's total, so a uniform panel is the natural starting
        # point; initializing at 0 would start λ a factor of ngenes too high.
        self.gene_bias = nn.Parameter(torch.full((ngenes,), -np.log(ngenes)))

        # background expression rates
        self.bg_rates = nn.Parameter(torch.zeros(ngenes))

        # negative-binomial overdispersion parameters
        self.log_r = nn.Parameter(torch.full((ngenes,), 2.0))

        # regression coefficient surrogate posterior parameters
        self.β_μ = nn.Parameter(torch.full((ncovariates, ngenes), 0.0))
        self.β_logσ = nn.Parameter(torch.full((ncovariates, ngenes), -2.0))

        # The drift direction is FIXED (a buffer, so it follows .to(device) and is
        # not optimized); only its per-column scale κ is fitted. One scalar per
        # design column, initialized at 0 so the fit starts from the uncorrected
        # solution and κ has to earn its way off it.
        # A prior on κ, and WHY it is worth having one. The likelihood is EXACTLY
        # invariant along (β + c*u, κ - c) -- the drift enters the mean only through
        # β + κ*u -- so the split is chosen entirely by the priors, not fitted. With
        # κ unpenalized that choice falls wholly on β_prior_σ, whose right value
        # depends on the panel and the effect-size distribution and so does not
        # transfer between datasets.
        #
        # κ transfers much better. Fitted per design group against measured damage it
        # sits at mean -1.46, sd 1.04 over 17 groups spanning breast (313 genes) and
        # Atera (16.7k-gene WTA) -- simple-seg-sim DESIGN_v2 §57. It also has an
        # external check no prior on β has: compare the fitted κ against the prior on
        # any new dataset. So N(-1.5, 1) is weakly informative rather than tuned, and
        # it takes the arbitrariness off β_prior_σ.
        #
        # NOT derived from the mixing geometry, which was tried and fails: the
        # per-gene donor/self ratio the first-order theory calls for is a ratio of two
        # tiny noisy numbers, and using it pointwise destroys the covariate (median R²
        # 0.24 -> 0.05 breast, 0.22 -> 0.01 Atera). Group-level geometry anchors
        # correlate only +0.33 with κ. The fitted scalar is doing real variance
        # reduction, not merely absorbing ignorance.
        self.κ_prior_μ = κ_prior_μ
        self.κ_prior_σ = κ_prior_σ
        if cellspace is not None:
            # frac is rebuilt per minibatch from these; materializing it would be
            # 3.0 GB on the Atera panel. One kappa, as for the drift term.
            for nm, arr, dt in (("cs_M", cellspace["M"], torch.float32),
                                ("cs_sw", cellspace["sw"], torch.float32),
                                ("cs_own", cellspace["own"], torch.long),
                                ("cs_P", cellspace["P"], torch.float32),
                                ("cs_gene", cellspace["gene_index"], torch.long)):
                self.register_buffer(nm, torch.as_tensor(
                    np.ascontiguousarray(arr)).to(dt).clone())
            self.κ_cs = nn.Parameter(torch.zeros(()))
            self.cellspace_percluster = cellspace_percluster
            self.cellspace_shrink_σ = cellspace_shrink_σ
            if cellspace_percluster:
                self.δ_cs = nn.Parameter(
                    torch.zeros(int(cellspace["P"].shape[0])))
        if drift is not None:
            # np.ascontiguousarray: a reindexed DataFrame's array can be read-only,
            # which torch warns about and would make the buffer's writability
            # undefined.
            self.register_buffer(
                "u", torch.as_tensor(np.ascontiguousarray(drift),
                                     dtype=torch.float32).clone())
            self.κ = nn.Parameter(torch.zeros(ncovariates))


    def forward(self, batch: RegressionBatch):
        if self.training:
            β_σ = torch.exp(self.β_logσ.clamp(max=4.0))
            β = self.β_μ + torch.randn_like(β_σ) * β_σ
        else:
            β = self.β_μ

        # regression
        # TODO: if this proves to be too inflexible, we may have to revive our old scheme of encoding
        # some amount of extra per-cell variation using a VAE term.
        #
        # log_size is a fixed offset, so exp(gene_bias) is a gene's share of a
        # cell's total and λ is on the absolute count scale, which is what the
        # mixing below needs in order to redistribute molecules correctly.
        # δ = κ*u is the contamination bias in coefficient space. It enters the
        # MEAN, so the fit reproduces the observed (biased) coefficients, while β
        # -- what is reported -- is what is left after it. β is penalized and κ is
        # not, which is what makes the split identifiable; see Regression.__init__.
        β_eff = β + self.κ.unsqueeze(1) * self.u if self.include_drift else β
        η = batch.design @ β_eff + self.gene_bias + batch.log_size.unsqueeze(1)
        if self.include_cellspace:
            κc = self.κ_cs
            if self.cellspace_percluster:
                κc = (self.κ_cs + self.δ_cs[self.cs_own[batch.nodes]]).unsqueeze(1)
            η = η + κc * self.cellspace_frac(batch)
        λ = torch.exp(η)

        λ_obs = self.mix(λ, batch)

        # Senders-only nodes exist to supply λ for the mixing; only receivers are
        # modeled, so the likelihood is over the leading block of the batch.
        X_obs = batch.X[: batch.nreceivers, :]

        # loss
        ll = self.negbinom_likelihood(λ_obs, X_obs)

        # β is a global parameter but the likelihood only covers this batch, so
        # the KL is down-weighted by the batch's share of the data. Summed over
        # an epoch these weights come to exactly 1, i.e. the prior is applied
        # once per pass. Without this it is applied once per *batch*, which is
        # ncells/batch_size times too strong.
        kl = self.β_kl(β) * (batch.nreceivers / self.ncells)
        # κ is a MAP parameter, not variational, so its prior is a plain penalty --
        # weighted like the KL so that a pass over the data applies it exactly once.
        if self.include_cellspace and self.cellspace_percluster:
            # Partial pooling on the per-cluster deviations, weighted like the KL so
            # one pass applies it exactly once.
            kl = kl + ((self.δ_cs ** 2).sum()
                       / (2.0 * self.cellspace_shrink_σ ** 2)) * (
                           batch.nreceivers / self.ncells)
        if self.include_drift and self.κ_prior_σ is not None:
            kl = kl + (((self.κ - self.κ_prior_μ) ** 2).sum()
                       / (2.0 * self.κ_prior_σ ** 2)) * (batch.nreceivers / self.ncells)

        return -ll.sum() + kl

    def cellspace_frac(self, batch: RegressionBatch) -> Tensor:
        """frac[i, g] for this batch's nodes, [nnodes, ngenes].

            F = M_i @ P          what the donor neighbourhood expresses
            O = sw_i * P[own_i]  what the cell's own cluster expresses
            frac = F / (F + O)

        Rebuilt rather than stored: the full field is 3.0 GB on Atera, while a
        2048-node batch is ~137 MB. Genes absent from the field get 0, which makes
        their correction a no-op rather than a NaN.
        """
        idx = batch.nodes
        F = self.cs_M[idx] @ self.cs_P                     # [nnodes, ngenes_field]
        O = self.cs_sw[idx].unsqueeze(1) * self.cs_P[self.cs_own[idx]]
        frac = F / (F + O).clamp(min=1e-9)
        g = self.cs_gene
        out = torch.zeros((frac.shape[0], g.shape[0]), dtype=frac.dtype,
                          device=frac.device)
        keep = g >= 0
        out[:, keep] = frac[:, g[keep]]
        return out

    def β_kl(self, β: Tensor) -> Tensor:
        """KL(q(β) || prior), summed over all covariates and genes.

        `β` is the reparameterized draw the likelihood used this step; it is read
        only by the Cauchy branch, which has no closed form.

        NORMAL: KL(q || N(0, β_prior_σ²)), exact.

        CAUCHY: KL(q || Cauchy(0, β_prior_σ)), split as -H(q) - E_q[log p]. The
        entropy is exact; the cross-entropy is a ONE-SAMPLE estimate at the draw
        the likelihood already made. A second draw would add variance without
        information, and the gradient through β is unbiased either way. The point
        of the heavy tail: the penalty gradient is 2β/(γ² + β²), which pulls like
        2β/γ² near zero but decays like 2/β in the tail, so a scale tight enough
        to shrink the nulls does not also attenuate a real log-FC -- the coupling
        that refuted `β_prior_σ = 0.1` (simple-seg-sim DESIGN_v2 §62).

        In eval mode β is β_μ and the Cauchy estimate degenerates to a point
        evaluation; nothing calls this outside training.
        """
        β_logσ = self.β_logσ.clamp(max=4.0)
        if self.β_prior == "cauchy":
            neg_entropy = -(0.5 * np.log(2.0 * np.pi * np.e) + β_logσ)
            cross = np.log(np.pi * self.β_prior_σ) + torch.log1p(
                (β / self.β_prior_σ) ** 2)
            return (neg_entropy + cross).sum()
        prior_var = self.β_prior_σ**2
        return (
            np.log(self.β_prior_σ)
            - β_logσ
            + (torch.exp(2.0 * β_logσ) + self.β_μ**2) / (2.0 * prior_var)
            - 0.5
        ).sum()

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

        # Background is the remaining mixture component. bg_weight[i] is the
        # fraction of cell i's molecules that came from background, so this has
        # to be scaled by the cell's own size and bg_rates is a profile over
        # genes rather than an absolute rate.
        bg = batch.bg_weight[:nr] * torch.exp(batch.log_size[:nr])
        return λ_obs + bg.unsqueeze(1) * torch.softmax(self.bg_rates, dim=-1)

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
