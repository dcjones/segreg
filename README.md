# Segmentation-aware Regression

`segreg` implements log-linear Poisson / negative-binomial regression for spatial
transcriptomics, intended for tasks like differential expression testing.

Differential expression in this data is often confounded by **segmentation
error**: transcripts leak between neighboring cells, so an apparent expression
difference may just reflect where the segmentation boundary happened to fall.
`segreg` tries to reduce this by adding a term to the regression that explains
away effects attributable to segmentation error. For this to work it leans on
per-cell uncertainty estimates produced by
[proseg](https://github.com/dcjones/proseg), an MCMC-based segmentation tool.

> **Status:** This is a research project under active, rapidly changing
> development. Interfaces are not stable and results should be treated as
> experimental.

## Requirements

- **Python ≥ 3.13**
- [**uv**](https://docs.astral.sh/uv/) for dependency management (see below —
  all commands are run through `uv`)
- A **CUDA GPU is strongly recommended.** Training runs on CPU (it is selected
  automatically when no GPU is present) but is much slower.
- [**proseg**](https://github.com/dcjones/proseg) — required only for the
  segmentation-aware correction (`include_diffusion=True`). Without it you can
  still run a plain regression on any AnnData/SpatialData object.

## Installation

`segreg` is not published on PyPI; install it from a clone.

```bash
git clone https://github.com/dcjones/segreg.git
cd segreg
uv sync
```

`uv sync` creates a `.venv/` and installs everything pinned in `uv.lock`. Run
code in that environment with `uv run`, e.g. `uv run python my_script.py`. There
is no need to activate the virtualenv manually.

To also install the optional debugging/notebook extras (marimo, umap-learn):

```bash
uv sync --group debug
```

## Preparing input data

`segreg` operates on an [AnnData](https://anndata.readthedocs.io/) or
[SpatialData](https://spatialdata.scverse.org/) object.

For the segmentation-aware correction, the input must be **proseg output**,
which carries the per-cell inflow/outflow uncertainty estimates `segreg` needs.
Generating these currently requires a development branch of proseg — the **`em`**
branch is the best option right now:

```bash
git clone https://github.com/dcjones/proseg.git
cd proseg
git checkout em
cargo build --release
# proseg binary is now at target/release/proseg
```

## Usage

`RegressionModel` does all the work. Load your data and construct the model:

```python
import spatialdata
from segreg import RegressionModel

sdata = spatialdata.read_zarr("proseg-output.zarr")   # or: anndata.read_h5ad(...)

model = RegressionModel(
    sdata,
    formula="celltype * timepoint",
    include_diffusion=True,
)
```

- `include_diffusion=True` (default) enables the segmentation-error correction
  and **requires proseg input**. Set it to `False` to run a standard log-linear
  regression on generic AnnData/SpatialData.
- `formula` is a [patsy](https://patsy.readthedocs.io/en/latest/formulas.html)
  formula referencing columns of the `obs` table. It specifies which covariates
  to include and how to encode them; consult the patsy docs for details. The
  example `celltype * timepoint` regresses on cell-type and timepoint annotations
  and includes their interaction.

Fit the model (this may take a while):

```python
model.fit()
```

### Results

Regression coefficients are reported as bounds at a chosen credible interval.
For categorical covariates these can be read as bounds on log fold change:

```python
results_df = model.get_regression_coefficients(credible_interval=0.99)
```
