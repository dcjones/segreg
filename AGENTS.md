
# Purpose

This package is a research project to develop a segmentation-aware regression
model for spatial transcriptomics datasets.

The main idea is to take uncertainty information estimated by proseg, a MCMC
based cell segmentation tool I also developed, and to incorporate that into the
regression model implemented here with pytorch, trained as an amortized
(VAE-style) inference scheme.

The accompanying methods writeup lives in `paper/segreg-paper.typ` and should
be treated as the source of truth for notation and model formulation; the
implementation aims to track it.

# Organization

Dependencies are managed using uv, so run everything using that. For example, to
run some python code, use 'uv run python -c ...', not 'python -c ...'. Add
dependencies with 'uv add', and so on.

Some important components, all under `src/segreg/`:
  * `data.py`: Loading proseg/AnnData/SpatialData input (`load_proseg_data`),
    OLS-based initialization of regression coefficients (`ols_init_beta`), and
    `estimate_phi`, which computes the per-cell/gene retention fraction phi_cg
    from the true-count identity T = X + outflow - inflow.
  * `nn.py`: The regression VAE architecture (`Encoder`, `NodeDecoder`,
    `SegregVAE` — log-linear regression on a design matrix). The NMF-style factor
    model lives in `factorization.py`. Per the paper's segmentation-aware count
    model, outflow enters multiplicatively as a retention factor
    `exp(-alpha_g * phi_cg)` and inflow additively as `alpha_g * inflow_cg`.
    NOTE (empirically established, see the auto-memory `project_niche_de_prototype`):
    the retention/outflow term is inert-to-harmful, so `include_retention=False`
    is now the default; the additive inflow term does the decontamination. Several
    alpha variants exist as flags (`stochastic_alpha`, `component_alpha`,
    `separate_retention_alpha`) but are ineffective for the known spurious-DE
    failure mode and default off — see `next-step-generative-model.md`.
    The `likelihood` flag selects the count model: `"nb_mean"` (default) is the
    phenomenological NB on the combined mean `mu = lam + delta`; `"nb_conv"` is the
    paper's exact generative marginal (Option B) — signal `A ~ NB(lam, psi)` plus
    independent contamination `C ~ Poisson(delta)`, so the likelihood is their
    convolution and alpha is fixed at 1. `nb_conv` roughly halves the residual
    spurious-DE bias while preserving real-DE power (see the auto-memory
    `project_generative_decontamination_model`); the remaining overshoot is a
    mean-structure artifact, not a likelihood one. `"nb_mm"` is the moment-matched
    single-NB approximation to `nb_conv` — cheaper, but benchmarked WORSE (planted
    |bias| 0.41 vs 0.25, credible intervals ~2.5x too narrow, 0/7 covering truth vs
    4/7). Don't use it; the convolution's shape matters beyond its moments.
    The `contam_var` flag sets the variance of the contamination count `C`:
    `"poisson"` (default) `Var=delta`; `"proseg"` `Var=V_f` (binomial arm);
    `"proseg_rate"` `Var=delta+V_f` (NB arm, the Gamma-inflow-rate reading).
    **Both non-default modes were tested and rejected — keep `"poisson"`**; see
    the auto-memory `project_inflow_variance_rejected` for the numbers and the
    mechanism (sub-Poisson contamination caps the count that can be attributed to
    contamination at high counts, pushing it onto lambda near contaminating types).
  * `losses.py`: Loss terms (NB reconstruction, latent KL, size-factor prior,
    alpha prior) used by the training wrappers.
  * `training.py`: `SegregTrainingWrapper` / `FactorizationTrainingWrapper`,
    which combine a model forward pass with its loss terms for one batch.
  * `regression.py` / `factorization.py`: `RegressionModel` / `FactorizationModel`,
    the user-facing classes that own the data, build batches, run training
    loops, and expose corrected expression / latent representations /
    regression coefficients.

In the examples directory there is basically scratch code for quickly testing
things out along with example proseg output (`proseg-output.zarr`).

# What proseg generates (and how segreg uses it)

A proseg run is loaded as an AnnData / SpatialData table (`load_proseg_data`
checks for `uns["proseg_run"]`). Beyond the point-estimate count matrix, proseg
emits the segmentation-uncertainty quantities the whole project is built around:

  * `X` (cell x gene): point-estimate counts from proseg's point-estimate
    transcript assignments (sparse). Note `inflow <= X` elementwise — inflow is
    the contaminated *subset* of a cell's assigned counts, so ratios like
    `X/inflow` are floored at 1 and cannot diagnose over/under-estimation.
  * `layers["expected_inflow"]`, `layers["expected_outflow"]` (cell x gene):
    posterior-mean flows over the MCMC. Inflow_cg = expected transcripts of gene
    g assigned to cell c in the point estimate but that probably belong elsewhere
    (contamination in); outflow is the reverse (this cell's transcripts that
    leaked out). True-count identity: `T = X + outflow - inflow`; retention
    fraction `phi = outflow / T`.
  * `layers["expected_inflow_var"]`, `layers["expected_outflow_var"]`: posterior
    variances of those flows (how uncertain proseg is about each estimate).
  * `obs["component"]`: proseg's point-estimate expression-mixture assignment per
    cell (integer). Label-free, and *upstream* of the inflow estimate (it drives
    what proseg treats as "belonging" vs contamination), so it is a useful
    conditioning variable. Also `volume`/`surface_area` (morphology; `volume` is
    the default size factor), `centroid_{x,y,z}`, `original_cell_id` (join key),
    and on some datasets `het_inflow`/`het_outflow`/`heterotypic_uncertainty`.
  * `obsm["spatial"]` (cell centroids, for spatial-neighborhood construction) and
    `obsm["metagene_rates"]` (proseg's NMF metagene loadings per cell, ~50
    factors — a reduced expression representation); `varm["metagene_loadings"]`
    gives the gene x factor loadings (each metagene's identity).

# Evaluating changes

The deliverable is unconfounded DE, and the overriding goal is to **never report
spurious changes**. Evaluate accordingly — and prefer these over ad-hoc probes
(inferring "truth" through additional log-linear fits and reading point estimates
is fragile and has misled repeatedly):

  * **`simple-seg-sim/eval/score_de_panel.py` — the primary DE benchmark.** Scores
    EVERY (cell type, gene) pair from one fit (~4000 on the breast run) rather
    than ~10 hand-picked markers, against a per-gene truth calibrated from the
    simulator's ground-truth counts. Reports `FP = P(CI excludes 0 | truth null)`
    and `FN = P(CI covers 0 | truth real)`. Use this to judge any model change.
    Three things to know before reading its output, all learned the hard way:
    (a) truth must be labelled with QUASI-POISSON standard errors — with plain
    Poisson errors, overdispersion lets null genes leak into the effect set and
    the FN rate becomes meaningless (it read 0.87 when the truth was 0.17);
    (b) FN is NOT estimable on `out/full`, which plants only 7 coefficients — a
    config planting many more effects is needed before FN means anything;
    (c) nominal truths (`sign*strength`, 0 for should-be-null) are only valid
    under total-count normalization — `eval/truth_calibration.py` explains why,
    and turned up a "should-be-null" marker with a real +0.26 slope.
    This benchmark is what validates the project's central claim: disabling
    diffusion modeling quadruples the false-positive rate (0.14 -> 0.60).
  * `segreg.evaluation`: fit-free, un-gameable honest metrics. `marker_leakage`
    (scale-invariant, label-robust cross-type contamination ratio — the primary
    decontamination metric), plus `probe_separability`, `leiden_ari`, and the
    `evaluate()` harness. Uses mutually-exclusive cell-type markers.
  * Should-be-null coefficient check (primary for DE): on mutually-exclusive
    markers (a cell type does not express another type's markers), the regression
    coefficient's 95% credible interval should cover 0 — a call that excludes 0 is
    spurious. Quick local testbed: `examples/macrophage-prox-de/` (small NSCLC
    crop + `reference/cell-metadata.csv` with `celltype` and `has_tumor_neighbor`).
  * `spurious-de-benchmarks` repo (`/mnt/extra2/spurious-de-benchmarks/`): a
    Snakemake pipeline benchmarking DE methods on real spatial datasets across
    platforms (xenium/merfish/cosmx), deliberately configured to be susceptible
    to missegmentation confounding. Prepared datasets live under
    `results/datasets/{name}/` (each with proseg output); `results/benchmark/
    metrics.csv` aggregates. Use it to check whether a change reduces spurious DE
    without losing real DE. No artificial-contamination simulator exists yet, so
    evaluation is driven by assumed mutually-exclusive / cell-type-specific
    markers in real data.

Beware of the benchmark itself. Over one session three separate scoring defects
each produced a confident, wrong conclusion about the model: a collinear design
(r = 0.998 between exposure covariates) that split effects arbitrarily between
coefficients, truth values stated in a normalization the model does not use, and
Poisson standard errors that mislabelled noise as real effects. When a result
looks like a model defect, check the instrument first.

See `archive/next-step-generative-model.md` for the older open problem (a
spurious-DE degeneracy in the contamination-dominated regime). The current one is
narrower: the correction occasionally OVER-corrects into a larger spurious call
rather than removing one (Macrophages-1 PECAM1, truth +0.05, reads +0.71), and
this survives even with exactly correct inflow — see the auto-memories
`project_encoder_contamination_leak` and `project_panel_benchmark`.
