
from scipy.sparse import csr_matrix, coo_matrix
from torch import Tensor
import numpy as np
import numpy.typing as npt
import torch
from dataclasses import dataclass
from typing import cast


# TODO: So what do I actually need to fit the model?
#   1. X[batch_idx,:] of course
#   2. design[batch_idx,:] of course
#   3. A[batch_idx,:], but with indexes adjusted to point to a
#      Or maybe a senders and receivers arrays with local
#
@dataclass
class RegressionBatch:
    X: Tensor
    design: Tensor

    # graph structure with local within-batch indexes
    receivers_mask: Tensor # mask selecting just the receivers from X and design
    receivers: Tensor # nodes being modeled
    senders: Tensor # neighbors
    weights: Tensor # mixing amount
    bg_weight: Tensor # mixing weight with bg


class RegressionBatchLoader:
    X: csr_matrix
    A: csr_matrix
    bg_mix_rate: npt.NDArray[np.float32]
    design: npt.NDArray[np.float32]
    batch_size: int
    idx: npt.NDArray[np.int64]

    def __init__(self, X: csr_matrix, A: csr_matrix, bg_mix_rate: npt.NDArray[np.float32], design: npt.NDArray[np.float32], batch_size: int):
        self.X = X
        self.bg_mix_rate = bg_mix_rate
        self.A = A
        self.design = design
        self.batch_size = batch_size

        ncells = self.X.shape[0]
        self.idx = np.arange(ncells)

    def __iter__(self):
        # TODO: This is probably pretty sub-optimal. Rather we should prefer
        # to sample more connected chunks. We could take the spatial coordinates
        # and try to tile with a rough target batch_size.
        np.random.shuffle(self.idx)

        ncells = self.idx.shape[0]
        for fr in range(0, ncells, self.batch_size):
            to = min(ncells, fr + self.batch_size)

            print((fr, to))

            batch_receiver_idx = self.idx[fr:to]
            batch_receiver_set = set(batch_receiver_idx)

            A_batch = self.A[batch_receiver_idx,:].tocoo()
            assert isinstance(A_batch, coo_matrix)

            batch_idx_set = set(A_batch.row).union(A_batch.col)
            batch_idx = np.fromiter(batch_idx_set, dtype=int)
            batch_idx_map = {
                j: i for (i, j) in enumerate(batch_idx)
            }

            receiver_mask = np.asarray([i in batch_receiver_set for i in batch_idx], dtype=bool)
            receivers = np.fromiter((batch_idx_map[i] for i in A_batch.row), dtype=int)
            senders = np.fromiter((batch_idx_map[i] for i in A_batch.col), dtype=int)

            print(receivers)
            print(senders)

            yield RegressionBatch(
                X=torch.from_numpy(self.X[batch_idx,:].todense()),
                design=torch.from_numpy(self.design[batch_idx,:]),
                receivers_mask=torch.from_numpy(receiver_mask),
                receivers=torch.from_numpy(receivers),
                senders=torch.from_numpy(senders),
                weights=torch.from_numpy(A_batch.data),
                bg_weight=torch.from_numpy(self.bg_mix_rate[batch_idx])
            )
