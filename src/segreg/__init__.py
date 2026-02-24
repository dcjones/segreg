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
from torch_geometric.nn import GCNConv, Linear


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: layers
        pass

    def forward(self, x, edge_index):
        # TODO: apply layers
        #
        # Architerture is going to be something like:
        #
        # - Linear to get dense low dimensional representation
        # - GCN into softplus or exp to get μ and σ in the latent space
        #
        pass


class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, z):
        # TODO:
        # Two-header architecture to get
        #
        # Head 1: mlp to predict λ from z
        # Head 2: concatenate edge latent features with proseg prior (logit first?)
        #         mlp to sigmoid to get diffusion coeffficients
        #
        # Compute corrupted λ values to score observed x against
        pass


def objective():
    pass


class RegressionModel:
    X: csr_matrix
    data: Data
    design: DesignMatrix

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

        self.X = adata.X
        # TODO: cast to csr_matrix if we aren't already
        assert isinstance(self.X, csr_matrix)

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
                node_idx = batch.n_id.numpy()
                x_sub = self.X[node_idx]
                assert isinstance(x_sub, csr_matrix)

                x_sub_tensor = torch.sparse_csr_tensor(
                    torch.from_numpy(x_sub.indptr).to(torch.int32),
                    torch.from_numpy(x_sub.indices).to(torch.int32),
                    torch.from_numpy(x_sub.data).to(torch.float32),
                    size=x_sub.shape,
                ).to("cuda")

                print(x_sub_tensor)

                edge_index = batch.edge_index.to("cuda")

                # TODO: train step
