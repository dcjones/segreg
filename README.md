
# Segmentation-aware Regression

This implements a version of log-linear Poisson or negative Binomial regression for spatial transcriptomics that can be used for differential expression tests among other things.

Differential expression tests in this data are often confounded by segmentation error. Segreg tries to reduce this by adding a special term to the regression to explain away any effects that may be due to segmentation error. For this to work, it leans on uncertainty estimates made by [proseg](https://github.com/dcjones/proseg).

## Running segreg

All of this is in development and rapidly changing.

To generate the necessary uncertainty estimates, samples need to be processed with a development branch of proseg. Currently, the **em** branch in the [proseg
repo](https://github.com/dcjones/proseg/) is the best option.

```bash
git clone https://github.com/dcjones/proseg.git
cd proseg
git checkout em
cargo build --release
# proseg binary is now in: target/release/proseg
```


From segreg the `RegressionModel` class does all the work. This operates on either an AnnData or SpatialData object.


```python
from anndata import read_h5ad
from segreg import RegressionModel

adata = read_h5ad("dataset.h5ad")
model = RegressionModel(
    adata,
    formula="celltype * timepoint",
    include_diffusion=True,
)
```

If `include_diffusion=False` it runs a standard log-linear regression. If `True`, it tries to subtract out possible segmentation error.

How the regression is actually constructed is determined by `formula`, which references columns in the `obs` table in the AnnData object.

Segreg uses [patsy](https://patsy.readthedocs.io/en/latest/formulas.html) to parse these, so consult those docs for all the nitty gritty of how these work.

Broadly they specify which columns of `obs` should be included as covariates and how these covariates should be encoded. The formula here `celltype * timepoint` says to regress on the cell type annotations and timepoints and include interaction covariates between the two.


The model should then be fit, which may take a little while.
```python
model.fit()
```

Results currently are reported with respect to a particular credible interval. Primarily this reports upper and lower bounds for each regression coefficients. For categorical covariates this can be interpret as bounds on fold change.
```python
results_df = model.get_regression_coefficients(credible_interval=0.99)
```
