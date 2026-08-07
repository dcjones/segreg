
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch
from scipy.sparse import csr_matrix
from torch import Tensor


@dataclass
class RegressionBatch:
    # [nnodes, ngenes] counts for every node in the batch, receivers first
    X: Tensor

    # [nnodes, ncovariates] design matrix rows, receivers first
    design: Tensor

    # [nnodes] log total counts, used as a fixed regression offset
    log_size: Tensor

    # Number of receivers. Receivers occupy X[:nreceivers], senders-only
    # neighbors occupy X[nreceivers:].
    nreceivers: int

    # graph structure with local within-batch indexes
    receivers: Tensor  # nodes being modeled, one entry per edge
    senders: Tensor  # neighbors, one entry per edge
    weights: Tensor  # mixing amount, one entry per edge
    bg_weight: Tensor  # [nnodes] mixing weight with bg


@dataclass
class _TilePlan:
    """Everything about one tile that does not change from epoch to epoch.

    Tiles are fixed at construction and only their order is shuffled, so all of
    the index arithmetic below is identical on every pass. Doing it once and
    holding the result in pinned memory leaves the per-batch cost at a handful
    of async H2D copies.
    """

    nreceivers: int
    nnodes: int

    # X as (value, flat row-major index) pairs, scattered into a dense block
    # after the transfer. The block is only ~7.5% dense, so the sparse triple is
    # several times less data to move.
    x_values: Tensor
    x_flat_idx: Tensor

    design: Tensor
    log_size: Tensor
    bg_weight: Tensor

    receivers: Tensor
    senders: Tensor
    weights: Tensor


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
    log_size: npt.NDArray[np.float32]
    batch_size: int

    # Cells partitioned into spatially contiguous tiles of (at most) batch_size
    # cells. Tile order is shuffled each epoch, cells within a tile are not.
    tiles: list[npt.NDArray[np.int64]]

    # One per tile, in `tiles` order. Built once; see _TilePlan.
    plans: list[_TilePlan]

    def __init__(
        self,
        X: csr_matrix,
        A: csr_matrix,
        bg_mix_rate: npt.NDArray[np.float32],
        design: npt.NDArray[np.float32],
        log_size: npt.NDArray[np.float32],
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
        self.log_size = log_size
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

        # Scatter buffer for global -> local index mapping, reused across tiles
        # and always left as all -1.
        self._pos = np.full(ncells, -1, dtype=np.int64)

        self._A_indptr = A.indptr.astype(np.int64)
        self._X_indptr = X.indptr.astype(np.int64)
        self.plans = [self._plan(t) for t in self.tiles]

    def __len__(self) -> int:
        return len(self.plans)

    def _stage(self, x: npt.NDArray) -> Tensor:
        t = torch.from_numpy(np.ascontiguousarray(x))
        return t.pin_memory() if self.pin_memory else t

    def _plan(self, receiver_idx: npt.NDArray[np.int64]) -> _TilePlan:
        nreceivers = len(receiver_idx)
        ngenes = self.X.shape[1]

        # Edges out of the receivers, in global column space.
        pos, deg = _ragged_gather(self._A_indptr, receiver_idx)
        cols = self.A.indices[pos].astype(np.int64)
        weights = self.A.data[pos]

        # Local relabeling. Receivers are placed first so that they are a slice
        # rather than a mask, and stay in `receiver_idx` order.
        self._pos[receiver_idx] = np.arange(nreceivers)
        extra = np.unique(cols[self._pos[cols] < 0])
        self._pos[extra] = nreceivers + np.arange(len(extra))

        batch_idx = np.concatenate([receiver_idx, extra])
        receivers = np.repeat(np.arange(nreceivers), deg)
        senders = self._pos[cols]

        self._pos[batch_idx] = -1

        xpos, xdeg = _ragged_gather(self._X_indptr, batch_idx)
        x_rows = np.repeat(np.arange(len(batch_idx), dtype=np.int64), xdeg)
        x_flat_idx = x_rows * ngenes + self.X.indices[xpos].astype(np.int64)

        return _TilePlan(
            nreceivers=nreceivers,
            nnodes=len(batch_idx),
            x_values=self._stage(self.X.data[xpos].astype(np.float32)),
            x_flat_idx=self._stage(x_flat_idx),
            design=self._stage(self.design[batch_idx, :]),
            log_size=self._stage(self.log_size[batch_idx]),
            bg_weight=self._stage(self.bg_mix_rate[batch_idx]),
            receivers=self._stage(receivers),
            senders=self._stage(senders),
            weights=self._stage(weights),
        )

    def _to_device(self, x: Tensor) -> Tensor:
        if self.device is None:
            return x
        return x.to(self.device, non_blocking=self.pin_memory)

    def _materialize(self, plan: _TilePlan) -> RegressionBatch:
        ngenes = self.X.shape[1]

        values = self._to_device(plan.x_values)
        flat_idx = self._to_device(plan.x_flat_idx)
        X = torch.zeros((plan.nnodes * ngenes,), dtype=torch.float32, device=values.device)
        X.scatter_(0, flat_idx, values)

        return RegressionBatch(
            X=X.view(plan.nnodes, ngenes),
            design=self._to_device(plan.design),
            log_size=self._to_device(plan.log_size),
            nreceivers=plan.nreceivers,
            receivers=self._to_device(plan.receivers),
            senders=self._to_device(plan.senders),
            weights=self._to_device(plan.weights),
            bg_weight=self._to_device(plan.bg_weight),
        )

    def __iter__(self):
        for t in self.rng.permutation(len(self.plans)):
            yield self._materialize(self.plans[t])
