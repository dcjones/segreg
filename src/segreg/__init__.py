# Experimenting with writing this is pytorch instead to see how that would look.
from typing import cast

import numpy as np
import torch
from anndata import AnnData
from patsy import dmatrix
from patsy.design_info import DesignMatrix
from scipy.sparse import csr_matrix
from spatialdata import SpatialData
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

# from .model import RegressionModel


class RegressionModel:
    def __init__(
        self,
        data: SpatialData | AnnData,
        formula: str,
        batch_size: int | None = 4096,
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

        self.design = cast(DesignMatrix, dmatrix(formula, adata.obs))

        state_transitions = adata.obsp["state_transitions"]
        assert isinstance(state_transitions, csr_matrix)
        state_transitions_coo = state_transitions.tocoo()

        edge_index = np.stack(
            [state_transitions_coo.row, state_transitions_coo.col], axis=0
        )

        self.data = Data(
            edge_index=torch.tensor(edge_index, dtype=torch.int32),
            n_id=torch.arange(adata.n_obs),
            edge_attr=torch.tensor(state_transitions_coo.data, dtype=torch.float32),
            x=torch.tensor(np.asarray(self.design), dtype=torch.float32),
        )

        print(self.data)

    def fit(self, nepochs: int = 100, nneighbors: int = 10, batch_size: int = 1024):
        loader = NeighborLoader(
            self.data,
            num_neighbors=[nneighbors, nneighbors],
            batch_size=batch_size,
            input_nodes=None,
            shuffle=True,
        )

        for epoch in range(nepochs):
            print(f"Epoch {epoch}")
            for batch in loader:
                pass
