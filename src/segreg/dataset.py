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

    def __init__(self, X: csr_matrix, P: csr_matrix, batch_size: int):
        assert isinstance(X, csr_matrix)
        assert isinstance(P, csr_matrix)

        m, n = X.shape
        assert P.shape[0] == m and P.shape[1] == m

        X = X.astype(np.float32)
        self.n = n

        # Shuffle once at initialization
        idx = np.arange(m)
        np.random.shuffle(idx)

        # Compute batches along with their neighbors
        batch_idxs = []
        for fr in range(0, m, batch_size):
            to = min(fr + batch_size, m)

            batch_idx = idx[fr:to]

            # Use numpy operations to gather all neighbor indices at once
            # Get the CSR slices for all rows in batch_idx
            neighbor_indices = (
                np.concatenate(
                    [P.indices[P.indptr[i] : P.indptr[i + 1]] for i in batch_idx]
                )
                if len(batch_idx) > 0
                else np.array([], dtype=np.int32)
            )

            # Use np.unique to get sorted unique neighbor indices
            batch_senders_idx = np.unique(neighbor_indices)

            # Sort batch_idx and compute union with neighbors
            batch_idx_sorted = np.sort(batch_idx)
            batch_neighborhood_idx = np.union1d(batch_idx_sorted, batch_senders_idx)

            batch_idxs.append([batch_idx_sorted, batch_neighborhood_idx])

        # TODO: Feel like I may need to arrange these indices so we can easily subset
        # the main batch cells (exclude the neighborhood cells).

        # First pass: determine max nse across all batches
        x_max_nse = 0
        p_max_nse = 0
        max_nrows = 0
        for batch_idx, batch_neighborhood_idx in batch_idxs:
            x_sliced = X[batch_neighborhood_idx, :]
            # p_sliced = P[batch_neighborhood_idx, batch_neighborhood_idx] # This unfortunately seems to produce a dense matrix
            p_sliced = P[batch_neighborhood_idx, :][:, batch_neighborhood_idx]

            x_max_nse = max(x_max_nse, x_sliced.nnz)
            max_nrows = max(max_nrows, x_sliced.shape[0])
            p_max_nse = max(p_max_nse, p_sliced.nnz)

        # Second pass: Compute batch arrays
        self._batches = []
        for batch_idx, batch_neighborhood_idx in batch_idxs:
            x_sliced = X[batch_neighborhood_idx, :]
            x_nse_pad = x_max_nse - x_sliced.data.shape[0]
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
                        np.full(x_nrows_pad, x_sliced.shape[0], dtype=np.int32),
                    ]
                )
            else:
                x_indptr = x_sliced.indptr.copy()

            # p_sliced = P[batch_neighborhood_idx, batch_neighborhood_idx] # again, this produces a dense matrix
            p_sliced = P[batch_neighborhood_idx, :][:, batch_neighborhood_idx]
            p_nse_pad = p_max_nse - p_sliced.data.shape[0]
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
                        np.full(p_nrows_pad, p_sliced.shape[0], dtype=np.int32),
                    ]
                )
            else:
                p_indptr = p_sliced.indptr.copy()

            m_batch = x_sliced.shape[0]
            assert p_sliced.shape[0] == m_batch

            print(f"x_nrows_pad: {x_nrows_pad}")
            print(f"p_nrows_pad: {p_nrows_pad}")

            self._batches.append(
                (m_batch, x_data, x_indices, x_indptr, p_data, p_indices, p_indptr)
            )

        self.max_nrows = max_nrows
        self.n = n

    def __iter__(self):
        for (
            _m_batch,
            x_data,
            x_indices,
            x_indptr,
            p_data,
            p_indices,
            p_indptr,
        ) in self._batches:
            x_batch = BCSR(
                (
                    jnp.array(x_data),
                    jnp.array(x_indices, dtype=jnp.int32),
                    jnp.array(x_indptr, dtype=jnp.int32),
                ),
                shape=(self.max_nrows, self.n),
            )

            # TODO: I don't think this is quite right because
            #
            p_batch = BCSR(
                (
                    jnp.array(p_data),
                    jnp.array(p_indices, dtype=jnp.int32),
                    jnp.array(p_indptr, dtype=jnp.int32),
                ),
                # shape=(m_batch, m_batch),
                shape=(self.max_nrows, self.max_nrows),
            )

            yield x_batch, p_batch
