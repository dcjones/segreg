
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
  * `nn.py`: The VAE architecture (`Encoder`, `NodeDecoder`, `SegregBase`) and
    the two decoder heads built on it: `SegregVAE` (log-linear regression on a
    design matrix) and `SegregFactorizationVAE` (NMF-style factor model).
    Outflow enters multiplicatively as a retention factor
    `exp(-alpha_g * phi_cg)` and inflow enters additively as `alpha_g *
    inflow_cg`, per the paper's segmentation-aware count model; alpha_g is a
    learned per-gene coefficient shared between the two terms.
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
