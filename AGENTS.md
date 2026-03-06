
# Purpose

This package is a research project to develop a segmentation-aware regression
model for spatial transcriptomics datasets.

The main idea is to take uncertainty information estimated by proseg, a MCMC
based cell segmentation tool I also developed, and to incorporate that into the
regression model implemented here with jax.

# Organization

Dependencies are managed using uv, so run everything using that. For example, to
run some python code, use 'uv run python -c ...', not 'python -c ...'. Add
dependencies with 'uv add', and so on.

Some important components:
  * `Dataset` (src/segreg/dataset.py): Logic for sampling batches for training.
    This is tricky because we need to keep everything sparse while preserving
    static sizes to that jit compilation works.
  * `RegressionModel` (src/segreg/model.py): The primary general purpose
    regression model. This is in some ways a straightforward Poisson regression
    model, but with the added layer of complexity that we allow some diffusion
    of signal between neighboring cells.

In the examples directory there is basically scratch code for quickly testing
things out along with an example proseg output in 'proseg-output.zarr'.
