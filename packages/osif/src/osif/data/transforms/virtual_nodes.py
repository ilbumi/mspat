"""Transforms for adding virtual nodes to the protein graph."""

import torch
from torch_cluster import grid_cluster, radius
from torch_geometric.data import Data
from torch_geometric.nn import MeanAggregation
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.transforms import BaseTransform

from osif.data.utils.protein.atom_tensors import AtomTensors
from osif.data.utils.protein.vocabs import ATOM_NAMES_TO_INDEX, DUMMY_TOKEN, RESIDUE_TO_INDEX, AtomPropertyName


class AddIntermediateVirtualNodes(BaseTransform):
    """Adds virtual nodes to the graph halfway between real nodes.

    This transform does not construct edges, use RadiusGraphTransform after this if you need them.
    """

    def __init__(
        self,
        edge_length: float = 4.1,
        clash_cutoff: float = 2.0,
        merge_cutoff: float = 1.0,
    ):
        """Initialize the transform.

        Args:
            edge_length (float, optional): half of the grid length. Defaults to 4.1.
            clash_cutoff (float, optional): cutoff for clashing nodes. Nodes closer than
                this to real nodes will be removed from the grid. Defaults to 2.0.
            merge_cutoff (float, optional): cutoff for merging virtual nodes. Nodes closer than`

        """
        self.edge_length = edge_length
        self.clash_cutoff = clash_cutoff
        self.merge_cutoff = merge_cutoff
        self._aggr = MeanAggregation()

    def forward(self, data: Data) -> Data:
        r"""Add virtual nodes to the graph.

        N -- D -- CA.

        Args:
            data (Data): The data object wit pos and labels (see format in ProteinAtomsDataset).

        Returns:
            Data: The data object with added virtual nodes.

        """
        assert data.pos is not None  # noqa: S101
        assert data.labels is not None  # noqa: S101
        assert data.y is not None  # noqa: S101
        assert data.query_mask is not None  # noqa: S101
        indices_for_virtual_nodes_creation = radius(
            data.pos,
            data.pos,
            self.edge_length * 2,
            max_num_neighbors=128,
        )
        indices_for_virtual_nodes_creation = indices_for_virtual_nodes_creation[
            :,
            indices_for_virtual_nodes_creation[0] < indices_for_virtual_nodes_creation[1],
        ]  # remove self loops and duplicates

        virtual_nodes_pos = (
            data.pos[indices_for_virtual_nodes_creation[0]] + data.pos[indices_for_virtual_nodes_creation[1]]
        ) / 2
        distances = torch.cdist(virtual_nodes_pos, data.pos)
        virtual_nodes_pos = virtual_nodes_pos[distances.min(dim=1).values > self.clash_cutoff]
        if virtual_nodes_pos.shape[0] > 1:
            # merge close virtual nodes
            cluster_ids = consecutive_cluster(
                grid_cluster(
                    virtual_nodes_pos,
                    torch.tensor(
                        [self.merge_cutoff] * virtual_nodes_pos.shape[-1],
                        dtype=virtual_nodes_pos.dtype,
                        device=virtual_nodes_pos.device,
                    ),
                )
            )[0]
            virtual_nodes_pos = self._aggr(virtual_nodes_pos, cluster_ids)
        virtual_nodes_labels = AtomTensors.construct_labels(
            {
                AtomPropertyName.chain: [-1 for _ in range(len(virtual_nodes_pos))],
                AtomPropertyName.resindex: [-1 for _ in range(len(virtual_nodes_pos))],
                AtomPropertyName.residue_type: [RESIDUE_TO_INDEX[DUMMY_TOKEN] for _ in range(len(virtual_nodes_pos))],
                AtomPropertyName.atom_type: [ATOM_NAMES_TO_INDEX[DUMMY_TOKEN] for _ in range(len(virtual_nodes_pos))],
            }
        )

        data.pos = torch.cat([data.pos, virtual_nodes_pos], dim=0)
        data.labels = torch.cat([data.labels, virtual_nodes_labels], dim=0)
        data.y = torch.cat([data.y, virtual_nodes_labels[:, AtomPropertyName.residue_type]], dim=0)
        data.query_mask = torch.cat(
            [
                data.query_mask,
                torch.zeros(virtual_nodes_pos.shape[0], dtype=torch.bool),
            ],
            dim=0,
        )
        if "num_nodes" in data:
            data.num_nodes = len(data.pos)
        return data

    def __repr__(self):
        return f"{self.__class__.__name__}(edge_length={self.edge_length}, clash_cutoff={self.clash_cutoff})"
