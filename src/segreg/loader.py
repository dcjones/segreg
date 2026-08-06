
from scipy.sparse import csr_matrix
from torch import Tensor
import numpy as np
import numpy.typing as npt
import torch
from dataclasses import dataclass


@dataclass
class RegressionBatch:
    # [nnodes, ngenes] counts for every node in the batch, receivers first
    X: Tensor

    # [nnodes, ncovariates] design matrix rows, receivers first
    design: Tensor

    # Number of receivers. Receivers occupy X[:nreceivers], senders-only
    # neighbors occupy X[nreceivers:].
    nreceivers: int

    # graph structure with local within-batch indexes
    receivers: Tensor  # nodes being modeled, one entry per edge
    senders: Tensor  # neighbors, one entry per edge
    weights: Tensor  # mixing amount, one entry per edge
    bg_weight: Tensor  # [nnodes] mixing weight with bg


def _ragged_gather(
    indptr: npt.NDArray[np.int64], rows: npt.NDArray[np.int64]
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Positions of every nonzero entry in `rows`, and each row's degree.

    Equivalent to `np.concatenate([np.arange(indptr[i], indptr[i+1]) for i in rows])`
    but without the per-row python loop.
    """
    starts = indptr[rows]
    deg = indptr[rows + 1] - starts
    total = int(deg.sum())
    offsets = np.cumsum(deg) - deg
    pos = np.arange(total) - np.repeat(offsets, deg) + np.repeat(starts, deg)
    return pos, deg


def _morton_order(spatial: npt.NDArray[np.floating]) -> npt.NDArray[np.int64]:
    """Z-order (Morton) sort of 2D points, so that adjacent entries are spatial neighbors."""
    xy = np.asarray(spatial, dtype=np.float64)[:, :2]
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    q = (((xy - lo) / span) * (2**16 - 1)).astype(np.uint64)

    def interleave(v: npt.NDArray[np.uint64]) -> npt.NDArray[np.uint64]:
        v = (v | (v << np.uint64(16))) & np.uint64(0x0000FFFF0000FFFF)
        v = (v | (v << np.uint64(8))) & np.uint64(0x00FF00FF00FF00FF)
        v = (v | (v << np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
        v = (v | (v << np.uint64(2))) & np.uint64(0x3333333333333333)
        v = (v | (v << np.uint64(1))) & np.uint64(0x5555555555555555)
        return v

    morton = interleave(q[:, 0]) | (interleave(q[:, 1]) << np.uint64(1))
    return np.argsort(morton).astype(np.int64)


class RegressionBatchLoader:
    X: csr_matrix
    A: csr_matrix
    bg_mix_rate: npt.NDArray[np.float32]
    design: npt.NDArray[np.float32]
    batch_size: int

    # Cells partitioned into spatially contiguous tiles of (at most) batch_size
    # cells. Tile order is shuffled each epoch, cells within a tile are not.
    tiles: list[npt.NDArray[np.int64]]

    def __init__(
        self,
        X: csr_matrix,
        A: csr_matrix,
        bg_mix_rate: npt.NDArray[np.float32],
        design: npt.NDArray[np.float32],
        batch_size: int,
        spatial: npt.NDArray[np.floating] | None = None,
        seed: int | None = None,
        device: torch.device | str | None = None,
        pin_memory: bool = True,
    ):
        self.X = X
        self.A = A
        self.bg_mix_rate = bg_mix_rate
        self.design = design
        self.batch_size = batch_size
        self.device = torch.device(device) if device is not None else None
        self.rng = np.random.default_rng(seed)

        # Only worth pinning when we are actually going to transfer.
        self.pin_memory = pin_memory and self.device is not None and self.device.type == "cuda"

        ncells = X.shape[0]

        # Batching in spatial order keeps a batch's neighbors mostly inside the
        # batch, which is the difference between gathering ~10.7 and ~1.4 nodes
        # per modeled cell.
        order = _morton_order(spatial) if spatial is not None else np.arange(ncells, dtype=np.int64)
        self.tiles = [order[fr : fr + batch_size] for fr in range(0, ncells, batch_size)]

        # Scatter buffer for global -> local index mapping, reused across
        # batches and always left as all -1.
        self._pos = np.full(ncells, -1, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.tiles)

    def _to_device(self, x: Tensor) -> Tensor:
        if self.pin_memory:
            x = x.pin_memory()
        if self.device is not None:
            x = x.to(self.device, non_blocking=self.pin_memory)
        return x

    def _gather_X(self, batch_idx: npt.NDArray[np.int64]) -> Tensor:
        """Gather batch rows of X, transferring as CSR and densifying on the device.

        The batch block is ~7.5% dense, so shipping the sparse triple and
        expanding it after the transfer moves several times less data.
        """
        nrows = len(batch_idx)
        pos, deg = _ragged_gather(self.X.indptr.astype(np.int64), batch_idx)

        values = self._to_device(torch.from_numpy(self.X.data[pos].astype(np.float32)))
        cols = self._to_device(torch.from_numpy(self.X.indices[pos].astype(np.int64)))

        # Row indices are expanded on the device from the (much smaller) degrees
        # rather than transferred as one entry per nonzero.
        deg_t = self._to_device(torch.from_numpy(deg))
        rows = torch.repeat_interleave(
            torch.arange(nrows, device=deg_t.device), deg_t
        )

        X = torch.zeros((nrows, self.X.shape[1]), dtype=torch.float32, device=values.device)
        X[rows, cols] = values
        return X

    def __iter__(self):
        for t in self.rng.permutation(len(self.tiles)):
            receiver_idx = self.tiles[t]
            nreceivers = len(receiver_idx)

            # Edges out of the receivers, in global column space.
            pos, deg = _ragged_gather(self.A.indptr.astype(np.int64), receiver_idx)
            cols = self.A.indices[pos].astype(np.int64)
            weights = self.A.data[pos]

            # Local relabeling. Receivers are placed first so that they are a
            # slice rather than a mask, and stay in `receiver_idx` order.
            self._pos[receiver_idx] = np.arange(nreceivers)
            extra = np.unique(cols[self._pos[cols] < 0])
            self._pos[extra] = nreceivers + np.arange(len(extra))

            batch_idx = np.concatenate([receiver_idx, extra])
            receivers = np.repeat(np.arange(nreceivers), deg)
            senders = self._pos[cols]

            self._pos[batch_idx] = -1

            yield RegressionBatch(
                X=self._gather_X(batch_idx),
                design=self._to_device(torch.from_numpy(self.design[batch_idx, :])),
                nreceivers=nreceivers,
                receivers=self._to_device(torch.from_numpy(receivers)),
                senders=self._to_device(torch.from_numpy(senders)),
                weights=self._to_device(torch.from_numpy(weights)),
                bg_weight=self._to_device(torch.from_numpy(self.bg_mix_rate[batch_idx])),
            )
