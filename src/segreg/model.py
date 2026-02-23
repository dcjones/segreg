import jax.numpy as jnp
import numpy as np
from anndata import AnnData
from jax.experimental.sparse import BCSR, bcsr_extract
from scipy.sparse import csr_matrix


class RegressionModel:
    def __init__(
        self,
        adata: AnnData,
        formula: str,
        batch_size: int | None = 4096,
        nepochs: int = 100,
    ):
        m, n = adata.shape

        if batch_size is None:
            batch_size = m

        # TODO: convert if this is not a csr_matrix
        assert isinstance(adata.X, csr_matrix)

        # TODO: we need to decide what and where the proseg uncertainty output is
        dataset = Dataset(adata.X, P, batch_size)

        # Let's just check that we can iterate batches
        for x_batch, p_batch in dataset:
            print(x_batch.shape, p_batch.shape)

        # TODO: everything, all the training and such.
        pass

    def fit(self):
        pass
