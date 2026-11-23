import warnings
from typing import cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from anndata import AnnData
from patsy import dmatrix
from patsy.design_info import DesignMatrix
from scipy import stats
from scipy.sparse import coo_matrix, csr_matrix, issparse
from scipy.sparse import identity as sp_identity
from spatialdata import SpatialData
from torch import Tensor, nn

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

    def __init__(
        self,
        data: SpatialData | AnnData,
        formula: str,
        include_mixing: bool = True,
        include_cell_size: bool = False,
        mixing_scale: float = 1.0,
        mixing_range: tuple[float, float] | None = None,
    ):
        if isinstance(data, SpatialData):
            adata = data.tables["table"]
        elif isinstance(data, AnnData):
            adata = data
        else:
            raise TypeError("data must be an AnnData or SpatialData object")

        self.var_names = list(adata.var_names)
        self.include_mixing = include_mixing

        # In anndata X can be practically anything array like, but proseg always outputs csr matrices
        assert isinstance(adata.X, csr_matrix)
        self.X = adata.X

        # Construct design matrix
        # TODO: consider checking if the design has an intercept and
        # excluding the redundant gene_bias parameter in RegressionModel if so.
        design_df = dmatrix(formula, adata.obs, return_type="dataframe")
        self.design = cast(DesignMatrix, design_df)

        # Construct the neighbor mixing matrix
        state_transitions = adata.obsp["state_transitions"]
        assert isinstance(state_transitions, csr_matrix)

        assert "to_bg_trans_count" in adata.obs
        to_bg_trans_count = np.asarray(adata.obs["to_bg_trans_count"])

        total_transitions = (
            np.asarray(state_transitions.sum(axis=1)).squeeze() + to_bg_trans_count
        )

        # Avoid division by zero, typically in cases with entirely empty cells
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

        self._A_full = A.astype(np.float32)
        self._bg_full = (to_bg_trans_count / normalizer).astype(np.float32)
        if not 0.0 <= mixing_scale <= 1.0:
            raise ValueError(f"mixing_scale must be in [0, 1], got {mixing_scale}")
        if mixing_range is not None:
            lo, hi = mixing_range
            if not (0.0 <= lo <= mixing_scale <= hi <= 1.0):
                raise ValueError(
                    "mixing_range must satisfy 0 <= lo <= mixing_scale <= hi <= 1"
                )
            if not include_mixing:
                raise ValueError("mixing_range needs include_mixing=True")
        self.mixing_scale = float(mixing_scale)
        self.mixing_range = (
            None
            if mixing_range is None
            else (float(mixing_range[0]), float(mixing_range[1]))
        )
        self.A, self.bg_mix_rate = self._mixing_at(self.mixing_scale)

        counts = np.asarray(self.X.sum(axis=1)).squeeze()
        self.log_size = np.log(np.maximum(counts, 1)).astype(np.float32)
        self.include_cell_size = include_cell_size
        self.spatial = (
            np.asarray(adata.obsm["spatial"]) if "spatial" in adata.obsm else None
        )

    def fit(
        self,
        nepochs: int = 100,
        batch_size: int = 1024,
        lr: float = 0.01,
        β_prior: str = "cauchy",
        β_prior_σ: float = 1.0,
        size_prior_σ: float = 0.2,
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

        self.β_prior = β_prior
        self.size_prior_σ = size_prior_σ

        ncells, ngenes = self.X.shape
        model = RegressionModel(
            ncells,
            ngenes,
            self.design.shape[1],
            include_mixing=self.include_mixing,
            β_prior=β_prior,
            β_prior_σ=β_prior_σ,
            include_cell_size=self.include_cell_size,
            size_prior_σ=size_prior_σ,
        )
        if device is not None:
            model = model.to(device)

        param_device = next(model.parameters()).device

        # The parameters are small enough that a step's worth of unfused Adam
        # launches costs more than the rest of the batch put together.
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, fused=param_device.type == "cuda"
        )

        step = model.forward
        if compile:
            step = torch.compile(model.forward, dynamic=True)

        # snapshot β after 90% of nepochs to compute convergence statistics against the final β
        snap_at = max(int(nepochs * 0.9) - 1, 0)
        β_snap = None
        losses = []

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

            losses.append(float(total) / ncells)
            if epoch == snap_at:
                β_snap = model.β_μ.detach().clone()

            if verbose:
                # Per cell, so it stays comparable across batch sizes.
                print(f"epoch {epoch + 1}/{nepochs}: loss = {losses[-1]:.4f}")

        model.eval()
        self.model = model

        β_fin = model.β_μ.detach()
        tail = max(len(losses) // 10, 1)
        norm = float(torch.linalg.vector_norm(β_fin))
        move = (
            float(torch.linalg.vector_norm(β_fin - β_snap))
            if β_snap is not None
            else float("nan")
        )
        self.fit_diagnostics = {
            "nepochs": nepochs,
            "steps_per_epoch": int(np.ceil(ncells / batch_size)),
            "total_steps": int(nepochs * np.ceil(ncells / batch_size)),
            # Fraction of β's final magnitude that moved during the last 10% of
            # epochs. Not a hard threshold anywhere -- it is recorded so that
            # under-convergence is visible in a fit's own provenance instead of
            # requiring the experiment that found it.
            "β_rel_move_last10pct": move / norm if norm > 0 else float("nan"),
            "β_max_abs_move_last10pct": (
                float((β_fin - β_snap).abs().max())
                if β_snap is not None
                else float("nan")
            ),
            "β_l2": norm,
            "loss_first": losses[0] if losses else float("nan"),
            "loss_final": losses[-1] if losses else float("nan"),
            # Improvement still being made at the end, as a share of the whole run's
            # improvement. A converged fit has this near 0.
            "loss_tail_share": (
                (np.mean(losses[-2 * tail : -tail]) - np.mean(losses[-tail:]))
                / (losses[0] - losses[-1])
                if len(losses) >= 2 * tail and losses[0] != losses[-1]
                else float("nan")
            ),
            # Per-cell loss per epoch, for plotting the trajectory itself.
            "losses": losses,
        }
        if verbose:
            d = self.fit_diagnostics
            print(
                f"convergence: β moved {d['β_rel_move_last10pct']:.4f} of its norm "
                f"in the last 10% of epochs (max |Δβ| "
                f"{d['β_max_abs_move_last10pct']:.4f}); tail loss share "
                f"{d['loss_tail_share']:.4f}"
            )
        return model

    def posterior_sd(self, chunk: None | int = None) -> npt.NDArray[np.float64]:
        """Marginal posterior sd of β, [ncovariates, ngenes].

        Rather than use the surrogate posterior sd (`exp(β_logσ)`), this uses
        the Laplace approximation at the fitted β.
        """
        if self.model is None:
            raise Exception("fit() must be called before posterior_sd")

        m = self.model
        dev = next(m.parameters()).device
        B = m.β_μ.detach().double()
        gene_bias = m.gene_bias.detach().double()
        r = torch.exp(m.log_r.detach().double())
        bg_profile = torch.softmax(m.bg_rates.detach().double(), dim=-1)

        X = torch.as_tensor(np.asarray(self.design, dtype=np.float64), device=dev)
        log_size = torch.as_tensor(self.log_size.astype(np.float64), device=dev)

        # The fitted offset: this conditions on δ̂ rather than marginalizing it.
        if m.include_cell_size:
            log_size = log_size + m.δ_size.detach().to(dev).double()

        # The operator the model was fitted with, i.e. A at mixing_scale.
        A = torch.sparse_csr_tensor(
            torch.as_tensor(self.A.indptr, dtype=torch.int64, device=dev),
            torch.as_tensor(self.A.indices, dtype=torch.int64, device=dev),
            torch.as_tensor(self.A.data.astype(np.float64), device=dev),
            size=self.A.shape,
        )
        bg_weight = torch.as_tensor(
            self.bg_mix_rate.astype(np.float64), device=dev
        ) * torch.exp(log_size)

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
        if m.β_prior == "cauchy":
            prior_prec = 2.0 / (float(m.β_prior_σ) ** 2 + B**2)
        else:
            prior_prec = torch.full_like(B, 1.0 / float(m.β_prior_σ) ** 2)
        eye = torch.eye(ncov, dtype=torch.float64, device=dev)
        out = np.empty((ncov, ngenes), dtype=np.float64)

        # --- gene chunk size ------------------------------------------------------
        # Each chunk materializes a handful of dense [ncells, G, ncov] float64
        # tensors (the Jacobian, its weighted copy, the sparse product's input), so
        # the working set is ~5 of those. At 300k cells and 30 covariates a chunk of
        # 16 is ~5 GB, which does not fit beside a desktop on a 12 GB card. So by
        # default the chunk is sized to the memory actually free, and halved and
        # retried if a chunk still runs out. Chunking over genes is exact -- the
        # Hessian is block diagonal by gene -- so this changes nothing but speed.

        if dev.type == "cuda":
            torch.cuda.empty_cache()

        if chunk is None:
            chunk = 64
            if dev.type == "cuda":
                free, _ = torch.cuda.mem_get_info(dev)
                per_gene = ncells * ncov * 8 * 5
                chunk = int(min(chunk, max(1, (0.7 * free) // per_gene)))

        def laplace_cov(sl: slice) -> Tensor:
            """Laplace covariance H⁻¹ for one chunk of genes. Free of side effects,
            so a chunk that runs out of memory can simply be retried smaller."""
            G = sl.stop - sl.start
            η = X @ B[:, sl] + gene_bias[sl].unsqueeze(0) + log_size.unsqueeze(1)
            λ = torch.exp(η.clamp(max=30.0))
            del η
            if self.include_mixing:
                # [ncells, G, ncov], flattened so the sparse product is one call.
                d = (λ.unsqueeze(2) * X.unsqueeze(1)).reshape(ncells, G * ncov)
                J = (A @ d).reshape(ncells, G, ncov)
                del d
                μ = (A @ λ) + bg_weight.unsqueeze(1) * bg_profile[sl].unsqueeze(0)
            else:
                J = λ.unsqueeze(2) * X.unsqueeze(1)
                μ = λ
            w = r[sl].unsqueeze(0) / (μ * (r[sl].unsqueeze(0) + μ) + 1e-12)

            K = torch.einsum("ngc,ngd->gcd", J, J * w.unsqueeze(2))
            # [G, ncov, ncov] diagonal, since the precision now varies by
            # coefficient as well as by gene.
            H = K + prior_prec[:, sl].T.unsqueeze(2) * eye
            return torch.linalg.inv(H)

        start = 0
        while start < ngenes:
            sl = slice(start, min(start + chunk, ngenes))
            try:
                cov = laplace_cov(sl)
            except torch.OutOfMemoryError:
                if chunk == 1:
                    raise
                chunk = max(1, chunk // 2)
                torch.cuda.empty_cache()
                continue
            start = sl.stop
            sd = torch.diagonal(cov, dim1=1, dim2=2).clamp(min=0.0).sqrt()
            out[:, sl] = sd.T.cpu().numpy()

        return out

    def _mixing_at(self, scale: float) -> tuple[csr_matrix, npt.NDArray[np.float32]]:
        """(A_s, bg_s) at mixing scale s: (1 - s) I + s A and s·bg."""
        if scale == 1.0:
            return self._A_full, self._bg_full
        n = self._A_full.shape[0]
        A = (
            sp_identity(n, format="csr", dtype=np.float32) * (1.0 - scale)
            + self._A_full * scale
        ).tocsr()
        return A.astype(np.float32), (self._bg_full * scale).astype(np.float32)

    def range_interval(
        self,
        credible_interval: float,
        chunk: int = 64,
        max_iter: int = 30,
        tol: float = 1e-4,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """(lower, upper) bounds on β, [ncovariates, ngenes], over the mixing range.

        β is re-solved at each end of `mixing_range` and at `mixing_scale`, and the
        reported interval is the union of the three Laplace intervals, so a
        coefficient is significant only if it is significant across the range.

        Each refit holds the other fitted parameters fixed (gene_bias, r, the
        background profile, cell sizes) and finds β's MAP under the same prior by
        per-gene Fisher scoring with step halving, starting from the fitted β.
        """
        if self.model is None:
            raise Exception("fit() must be called before range_interval")
        if self.mixing_range is None:
            raise ValueError("range_interval needs a mixing_range")
        m = self.model

        dev = next(m.parameters()).device
        z = float(stats.norm.ppf(1.0 - (1.0 - credible_interval) / 2.0))
        B0 = m.β_μ.detach().double()
        gene_bias = m.gene_bias.detach().double()
        r = torch.exp(m.log_r.detach().double()).clamp(min=1e-3)
        bg_profile = torch.softmax(m.bg_rates.detach().double(), dim=-1)

        lo, hi = self.mixing_range
        settings = [lo, self.mixing_scale, hi]

        X = torch.as_tensor(np.asarray(self.design, dtype=np.float64), device=dev)
        log_size = torch.as_tensor(self.log_size.astype(np.float64), device=dev)
        if m.include_cell_size:
            log_size = log_size + m.δ_size.detach().double().to(dev)

        def mixing_tensors(scale):
            Aop, bg = self._mixing_at(scale)
            A = torch.sparse_csr_tensor(
                torch.as_tensor(Aop.indptr, dtype=torch.int64, device=dev),
                torch.as_tensor(Aop.indices, dtype=torch.int64, device=dev),
                torch.as_tensor(Aop.data.astype(np.float64), device=dev), size=Aop.shape)
            return A, torch.as_tensor(np.asarray(bg, dtype=np.float64), device=dev)

        cauchy = m.β_prior == "cauchy"
        γ = float(m.β_prior_σ)
        ncov, ngenes = B0.shape
        ncells = X.shape[0]
        eye = torch.eye(ncov, dtype=torch.float64, device=dev)
        lower = np.full((ncov, ngenes), np.inf)
        upper = np.full((ncov, ngenes), -np.inf)

        for scale in settings:
            A, bg_mix = mixing_tensors(scale)

            for start in range(0, ngenes, chunk):
                sl = slice(start, min(start + chunk, ngenes))
                y = self._counts_chunk(sl, dev)
                rg = r[sl].unsqueeze(0)

                def mean(B):
                    η = X @ B + gene_bias[sl].unsqueeze(0) + log_size.unsqueeze(1)
                    λ = torch.exp(η.clamp(max=30.0))
                    μ = A @ λ + (bg_mix * torch.exp(log_size)).unsqueeze(
                        1
                    ) * bg_profile[sl].unsqueeze(0)
                    return λ, μ

                def objective(B):
                    _, μ = mean(B)
                    nll = -(
                        y * torch.log(μ + 1e-8) - (y + rg) * torch.log(rg + μ + 1e-8)
                    ).sum(0)
                    if cauchy:
                        return nll + torch.log1p((B / γ) ** 2).sum(0)
                    return nll + (B**2).sum(0) / (2.0 * γ**2)

                def curvature(B):
                    """Ascent direction (score minus prior gradient) and the Fisher
                    information plus the prior's gradient-matched curvature."""
                    λ, μ = mean(B)
                    G = B.shape[1]
                    J = (
                        A
                        @ (λ.unsqueeze(2) * X.unsqueeze(1)).reshape(
                            ncells, G * ncov
                        )
                    ).reshape(ncells, G, ncov)
                    if cauchy:
                        pgrad, pprec = 2.0 * B / (γ**2 + B**2), 2.0 / (γ**2 + B**2)
                    else:
                        pgrad, pprec = B / γ**2, torch.full_like(B, 1.0 / γ**2)
                    w = rg / (μ * (rg + μ) + 1e-12)
                    grad = torch.einsum("ngc,ng->gc", J, w * (y - μ)) - pgrad.T
                    H = (
                        torch.einsum("ngc,ngd->gcd", J, J * w.unsqueeze(2))
                        + pprec.T.unsqueeze(2) * eye
                    )
                    return grad, H

                B = B0[:, sl].clone()
                nll = objective(B)
                for _ in range(max_iter):
                    grad, H = curvature(B)
                    step = (
                        torch.linalg.solve(H, grad.unsqueeze(2)).squeeze(2).T
                    )  # [ncov, G]
                    t = torch.ones(B.shape[1], dtype=torch.float64, device=dev)
                    for _ in range(20):  # per-gene step halving, on the objective alone
                        nll_new = objective(B + step * t)
                        bad = ~(nll_new <= nll + 1e-10)
                        if not bad.any():
                            break
                        t = torch.where(bad, t / 2, t)
                    t = torch.where(
                        bad, torch.zeros_like(t), t
                    )  # no descent found: stay
                    B = B + step * t
                    nll = torch.where(bad, nll, nll_new)
                    if float((step * t).abs().max()) < tol:
                        break
                _, H = curvature(B)
                sdv = (
                    torch.diagonal(torch.linalg.inv(H), dim1=1, dim2=2)
                    .clamp(min=0.0)
                    .sqrt()
                    .T
                )
                Bn, sdn = B.cpu().numpy(), sdv.cpu().numpy()
                lower[:, sl] = np.minimum(lower[:, sl], Bn - z * sdn)
                upper[:, sl] = np.maximum(upper[:, sl], Bn + z * sdn)
        return lower, upper

    def _counts_chunk(self, sl: slice, dev) -> Tensor:
        """Observed counts for a slice of genes, as a dense [ncells, G] tensor.

        Column-slicing a CSR matrix costs a pass over its nnz, and range_interval asks
        for ~n_genes/chunk slices (over a thousand on a WTA panel), so the CSC copy
        is built once and cached. It is the same nnz as self.X, not a densification.
        """
        X = self.X
        if issparse(X):
            csc = getattr(self, "_X_csc", None)
            if csc is None:
                csc = self._X_csc = X.tocsc()
            blk = np.asarray(csc[:, sl].todense(), dtype=np.float64)
        else:
            blk = np.asarray(X[:, sl], dtype=np.float64)
        return torch.as_tensor(blk, device=dev)

    def get_cell_sizes(self) -> pd.DataFrame:
        """The fitted per-cell size factor, one row per cell.

        `delta` is the learned deviation from `log(observed total)` and `scale` is
        exp of it, so scale 1 means the cell kept the fixed offset. `log_size_fixed`
        is that offset and `log_size_fitted` is what the model actually used.

        Read `delta` against `log_size_fixed`: a δ that is uncorrelated with the
        cell's total is the mixing correction the offset could not express, while a δ
        that tracks the total is the model disagreeing about depth itself, which can
        compete with the DE signal.
        """
        if self.model is None or not self.model.include_cell_size:
            raise Exception("no cell size factor on this fit")
        d = self.model.δ_size.detach().cpu().numpy().astype(np.float64)
        return pd.DataFrame(
            {
                "delta": d,
                "scale": np.exp(d),
                "log_size_fixed": self.log_size.astype(np.float64),
                "log_size_fitted": self.log_size.astype(np.float64) + d,
            }
        )

    def get_regression_coefficients(
        self, credible_interval: float | None = None
    ) -> pd.DataFrame:
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
            if self.mixing_range is not None:
                lo, hi = self.range_interval(credible_interval)
            else:
                β_σ = self.posterior_sd()
                z = stats.norm.ppf(1.0 - (1.0 - credible_interval) / 2.0)
                lo, hi = β_μ - z * β_σ, β_μ + z * β_σ
            df["Lower"] = lo.flatten(order="F")
            df["Upper"] = hi.flatten(order="F")
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

    def __init__(
        self,
        ncells: int,
        ngenes: int,
        ncovariates: int,
        include_mixing: bool = True,
        β_prior: str = "cauchy",
        β_prior_σ: float = 1.0,
        include_cell_size: bool = False,
        size_prior_σ: float = 0.2,
    ):
        super().__init__()

        self.include_mixing = include_mixing

        # A per-cell log-scale, learned as a deviation from log(observed total).
        self.include_cell_size = include_cell_size
        self.size_prior_σ = size_prior_σ
        if include_cell_size:
            self.δ_size = nn.Parameter(torch.zeros(ncells))

        # Needed to weight the KL against a minibatch's share of the likelihood.
        self.ncells = ncells

        if β_prior not in ("normal", "cauchy"):
            raise ValueError(f"β_prior must be 'normal' or 'cauchy', got {β_prior!r}")
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

    def forward(self, batch: RegressionBatch):
        if self.training:
            β_σ = torch.exp(self.β_logσ.clamp(max=4.0))
            β = self.β_μ + torch.randn_like(β_σ) * β_σ
        else:
            β = self.β_μ

        # regression
        log_size = batch.log_size
        if self.include_cell_size:
            log_size = log_size + self.δ_size[batch.nodes]
        η = batch.design @ β + self.gene_bias + log_size.unsqueeze(1)
        λ = torch.exp(η)

        λ_obs = self.mix(λ, batch, log_size=log_size)

        # Senders-only nodes exist to supply λ for the mixing; only receivers are
        # modeled, so the likelihood is over the leading block of the batch.
        X_obs = batch.X[: batch.nreceivers, :]

        # loss
        ll = self.negbinom_likelihood(λ_obs, X_obs)

        # Scale kl, since β is a global parameter, but here we are computing
        # loss for a single batch. Otherwise the strength of β's prior is not
        # properly scaled.
        kl = self.β_kl(β) * (batch.nreceivers / self.ncells)

        if self.include_cell_size:
            kl = kl + ((self.δ_size**2).sum() / (2.0 * self.size_prior_σ**2)) * (
                batch.nreceivers / self.ncells
            )

        return -ll.sum() + kl

    def β_kl(self, β: Tensor) -> Tensor:
        """KL(q(β) || prior), summed over all covariates and genes.

        `β` is the reparameterized draw the likelihood used this step; it is read
        only by the Cauchy branch, which has no closed form.

        NORMAL: KL(q || N(0, β_prior_σ²)), exact.

        CAUCHY: KL(q || Cauchy(0, β_prior_σ)), split as -H(q) - E_q[log p]. The
        entropy is exact; the cross-entropy is a ONE-SAMPLE estimate at the draw
        the likelihood already made. A second draw would add variance without
        information, and the gradient through β is unbiased either way.

        In eval mode β is β_μ and the Cauchy estimate degenerates to a point
        evaluation; nothing calls this outside training.
        """
        β_logσ = self.β_logσ.clamp(max=4.0)
        if self.β_prior == "cauchy":
            neg_entropy = -(0.5 * np.log(2.0 * np.pi * np.e) + β_logσ)
            cross = np.log(np.pi * self.β_prior_σ) + torch.log1p(
                (β / self.β_prior_σ) ** 2
            )
            return (neg_entropy + cross).sum()
        prior_var = self.β_prior_σ**2
        return (
            np.log(self.β_prior_σ)
            - β_logσ
            + (torch.exp(2.0 * β_logσ) + self.β_μ**2) / (2.0 * prior_var)
            - 0.5
        ).sum()

    def mix(
        self,
        λ: Tensor,
        batch: RegressionBatch,
        log_size: Tensor | None = None,
    ) -> Tensor:
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

        w = batch.weights

        # [nedges, ngenes], one row per (receiver, sender) pair
        contrib = w.unsqueeze(1) * λ[batch.senders, :]

        λ_obs = torch.zeros((nr, λ.shape[1]), dtype=λ.dtype, device=λ.device)
        λ_obs.index_add_(0, batch.receivers, contrib)

        # Background is the remaining mixture component. bg_weight[i] is the
        # fraction of cell i's molecules that came from background, so this has
        # to be scaled by the cell's own size and bg_rates is a profile over
        # genes rather than an absolute rate.
        ls = batch.log_size if log_size is None else log_size
        bg = batch.bg_weight[:nr] * torch.exp(ls[:nr])
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
