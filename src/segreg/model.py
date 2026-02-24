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

        if batch_size is None:
            batch_size = m

        # TODO: convert if this is not a csr_matrix
        assert isinstance(adata.X, csr_matrix)

        P = adata.obsp["state_transitions"]
        assert isinstance(P, csr_matrix)

        self._dataset = Dataset(adata.X, P, batch_size)

        # Let's just check that we can iterate batches
        for x_batch, p_batch in self._dataset:
            print(x_batch.shape, p_batch.shape)

    def fit(self):
        pass
