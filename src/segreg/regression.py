
from anndata import AnnData
from scipy.sparse import csr_matrix, coo_matrix
from spatialdata import SpatialData
from patsy import dmatrix
from patsy.design_info import DesignMatrix
from typing import cast
import numpy as np
import numpy.typing as npt
import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from .loader import RegressionBatchLoader


class RegressionModel:
    # count matrix point estimate
    X: csr_matrix

    # sparse mixing matrix between cell pairs. Rows sum to 1.
    A: csr_matrix
    bg_mix_rate: npt.NDArray[np.float32]

    design: DesignMatrix


    def __init__(self, data: SpatialData | AnnData, formula: str):
        if isinstance(data, SpatialData):
            adata = data.tables["table"]
        elif isinstance(data, AnnData):
            adata = data
        else:
            raise TypeError("data must be an AnnData or SpatialData object")

        # In anndata X can be practically anything array like, but proseg always outputs csr matrices
        assert isinstance(adata.X, csr_matrix)
        self.X = adata.X

        # Construct design matrix
        design_df = dmatrix(formula, adata.obs, return_type="dataframe")
        self.design = cast(DesignMatrix, design_df)

        # Construct the neighbor mixing matrix
        state_transitions = adata.obsp["state_transitions"]
        assert isinstance(state_transitions, csr_matrix)

        # TODO: Okay, where does from_bg_trans_count come into play? Do we actually need that?
        # assert "from_bg_trans_count" in adata.obs
        # from_bg_trans_count = np.asarray(adata.obs["from_bg_trans_count"])

        assert "to_bg_trans_count" in adata.obs
        to_bg_trans_count = np.asarray(adata.obs["to_bg_trans_count"])

        total_transitions = np.asarray(state_transitions.sum(axis=1)).squeeze() + to_bg_trans_count
        A = state_transitions / total_transitions[:, None]
        assert isinstance(A, coo_matrix)
        A = A.tocsr()
        assert isinstance(A, csr_matrix)
        self.A = A.astype(np.float32)

        self.bg_mix_rate = (to_bg_trans_count / total_transitions).astype(np.float32)

        # Construct Data object
        # ncells = self.X.shape[0]
        # self.data = Data(
        #     x=torch.from_numpy(np.asarray(self.design, dtype=np.float32)),
        #     edge_index=torch.from_numpy(
        #         np.stack([self.A.row, self.A.col]).astype(np.int64)
        #     ),
        #     edge_weight=torch.from_numpy(self.A.data),
        #     num_nodes=ncells,
        # )

    def fit(self, nepochs: int=1, batch_size: int=1024, batch_neighbors: int=10):
        loader = RegressionBatchLoader(
            self.X, self.A, self.bg_mix_rate, np.asarray(self.design, dtype=np.float32), batch_size
        )

        for epoch in range(nepochs):
            for batch in loader:
                print(batch)

                # TODO: slice into self.X
                # Should we pre-convert X to a sparse tensor?

                # TODO: training

    def regression_coefficients(self):
        pass
