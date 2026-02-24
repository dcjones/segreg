from typing import cast

import jax.numpy as jnp
import numpy as np
from anndata import AnnData
from jax.experimental.sparse import BCSR, bcsr_extract
from patsy import dmatrix
from patsy.design_info import DesignMatrix
from scipy.sparse import csr_matrix
from spatialdata import SpatialData

from .dataset import Dataset


class RegressionModel:
    def __init__(
        self,
        data: AnnData | SpatialData,
        formula: str,
        batch_size: int | None = 4096,
        nepochs: int = 100,
    ):

        if isinstance(data, AnnData):
            adata = data
        elif isinstance(data, SpatialData):
            adata = data.tables["table"]
        else:
            raise ValueError("data must be an AnnData or SpatialData object")

        if "proseg_run" not in adata.uns:
            raise ValueError("This is not a proseg spatialdata file")

        m, n = adata.shape

        design = cast(DesignMatrix, dmatrix(formula, adata.obs))
        # Convert design matrix to a plain numpy array
        D = np.asarray(design)

        if batch_size is None:
            batch_size = m

        # TODO: convert if this is not a csr_matrix
        assert isinstance(adata.X, csr_matrix)

        P = adata.obsp["state_transitions"]
        assert isinstance(P, csr_matrix)

        self._dataset = Dataset(adata.X, P, D, batch_size)

        # Let's just check that we can iterate batches
        for x_batch, p_batch, d_batch, mask in self._dataset:
            print(x_batch.shape, p_batch.shape, d_batch.shape, mask.shape)

    def fit(self):
        pass


# TODO:
# Now we move on to the really tricky part of trying write the model.
#
# The hard part is that we need to try to infer diffusion between every pair of
# cells. Since we are working with minibatches, we can't keep that in memory,
# and instead have to use some kind of amortized analysis. I'm really not sure
# what that looks like though. Some kind of graph neural network presumably.
#
# What is the input and output for such a thing though? Certainly it can't just
# be a MLP across neighbors, right?
#
# Questions:
#   - Do we want to use numpyro for this, or try to get away without it?
#   - Do we
#
# I think I just have to review graph neural networks. Surely there is a way to
# make edge predictions, right?
#
# Maybe I should just use pytorch since that has pytorch geometric and is maybe
# better suited to do GNNs.
#
# So there are GNNs that operate on edges, so we should be able to use something like
# that to predict
#
# But remember, we're also going to have to predict λ values for every cell as well. That's
# also hidden state that we can't assume we can store.
#
#
