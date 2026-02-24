from typing import cast

import jax.numpy as jnp
import numpy as np
from anndata import AnnData
from jax.experimental.sparse import BCSR, bcsr_extract
from scipy.sparse import csr_matrix
from spatialdata import SpatialData

# TODO:
# I think I precompute CSR batches, but along with neighbors.
# We need to keep track of indices of the main group of cells.


# TODO:
# This should also probably handle subsetting the observation matrix we
# are regressing on, right?


# TODO: Ok, maybe this part should be lower level and we have some different interface
# for


class Dataset:
    """
    The strategy here in to pre-compute batches, so we can iterate over them quickly. The two main complications are:
        1. We need batched cells to bring along all their neighbors so we can model diffusion.
        2. Jax's jit will recompile if the input array sizes change, so we have to pad these to always be the same length.
    """

    def __init__(self, X: csr_matrix, P: csr_matrix, D: np.ndarray, batch_size: int):
        assert isinstance(X, csr_matrix)
        assert isinstance(P, csr_matrix)
        assert isinstance(D, np.ndarray)

        m, n = X.shape
        assert P.shape[0] == m and P.shape[1] == m
        assert D.shape[0] == m

        X = X.astype(np.float32)
        P = P.astype(np.float32)
        D = D.astype(np.float32)
        self.n = n
        self.k = D.shape[1]

        # Shuffle once at initialization
        idx = np.arange(m)
        np.random.shuffle(idx)

        # Compute batches along with their neighbors
        batch_info = []
        for fr in range(0, m, batch_size):
            to = min(fr + batch_size, m)
            batch_idx = idx[fr:to]

            # Use numpy operations to gather all neighbor indices at once
            neighbor_indices = (
                np.concatenate(
                    [P.indices[P.indptr[i] : P.indptr[i + 1]] for i in batch_idx]
                )
                if len(batch_idx) > 0
                else np.array([], dtype=np.int32)
            )

            batch_senders_idx = np.unique(neighbor_indices)

            # To easily subset the main batch cells, we put them at the beginning of the neighborhood
            neighbors_exclusive = np.setdiff1d(batch_senders_idx, batch_idx)
            batch_neighborhood_idx = np.concatenate([batch_idx, neighbors_exclusive])

            batch_info.append((len(batch_idx), batch_neighborhood_idx))

        # First pass: determine max nse across all batches
        x_max_nse = 0
        p_max_nse = 0
        max_nrows = 0
        for n_target, batch_neighborhood_idx in batch_info:
            x_sliced = X[batch_neighborhood_idx, :]
            p_sliced = P[batch_neighborhood_idx, :][:, batch_neighborhood_idx]

            x_max_nse = max(x_max_nse, x_sliced.nnz)
            max_nrows = max(max_nrows, x_sliced.shape[0])
            p_max_nse = max(p_max_nse, p_sliced.nnz)

        # Second pass: Compute batch arrays
        self._batches = []
        for n_target, batch_neighborhood_idx in batch_info:
            x_sliced = X[batch_neighborhood_idx, :]
            x_nse_pad = x_max_nse - x_sliced.nnz
            x_nrows_pad = max_nrows - x_sliced.shape[0]

            if x_nse_pad > 0:
                x_data = np.concatenate(
                    [x_sliced.data, np.zeros(x_nse_pad, dtype=np.float32)]
                )
                x_indices = np.concatenate(
                    [x_sliced.indices, np.zeros(x_nse_pad, dtype=np.int32)]
                )
            else:
                x_data = x_sliced.data.copy()
                x_indices = x_sliced.indices.copy()

            if x_nrows_pad > 0:
                x_indptr = np.concatenate(
                    [
                        x_sliced.indptr,
                        np.full(x_nrows_pad, x_sliced.indptr[-1], dtype=np.int32),
                    ]
                )
            else:
                x_indptr = x_sliced.indptr.copy()

            p_sliced = P[batch_neighborhood_idx, :][:, batch_neighborhood_idx]
            p_nse_pad = p_max_nse - p_sliced.nnz
            p_nrows_pad = max_nrows - p_sliced.shape[0]

            if p_nse_pad > 0:
                p_data = np.concatenate(
                    [p_sliced.data, np.zeros(p_nse_pad, dtype=np.float32)]
                )
                p_indices = np.concatenate(
                    [p_sliced.indices, np.zeros(p_nse_pad, dtype=np.int32)]
                )
            else:
                p_data = p_sliced.data.copy()
                p_indices = p_sliced.indices.copy()

            if p_nrows_pad > 0:
                p_indptr = np.concatenate(
                    [
                        p_sliced.indptr,
                        np.full(p_nrows_pad, p_sliced.indptr[-1], dtype=np.int32),
                    ]
                )
            else:
                p_indptr = p_sliced.indptr.copy()

            d_sliced = D[batch_neighborhood_idx, :]
            if x_nrows_pad > 0:
                d_data = np.concatenate(
                    [
                        d_sliced,
                        np.zeros((x_nrows_pad, self.k), dtype=np.float32),
                    ],
                    axis=0,
                )
            else:
                d_data = d_sliced.copy()

            self._batches.append(
                (
                    n_target,
                    x_data,
                    x_indices,
                    x_indptr,
                    p_data,
                    p_indices,
                    p_indptr,
                    d_data,
                )
            )

        self.max_nrows = max_nrows
        self.n = n

    def __iter__(self):
        for (
            n_target,
            x_data,
            x_indices,
            x_indptr,
            p_data,
            p_indices,
            p_indptr,
            d_data,
        ) in self._batches:

            x_batch = BCSR(
                (
                    jnp.array(x_data),
                    jnp.array(x_indices, dtype=jnp.int32),
                    jnp.array(x_indptr, dtype=jnp.int32),
                ),
                shape=(self.max_nrows, self.n),
            )

            p_batch = BCSR(
                (
                    jnp.array(p_data),
                    jnp.array(p_indices, dtype=jnp.int32),
                    jnp.array(p_indptr, dtype=jnp.int32),
                ),
                shape=(self.max_nrows, self.max_nrows),
            )

            d_batch = jnp.array(d_data)

            # Mask identifying the target cells in the batch (the first n_target rows)
            mask = jnp.arange(self.max_nrows) < n_target

            yield x_batch, p_batch, d_batch, mask
