"""Torch OSIF model."""

import torch
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn.models.schnet import GaussianSmearing

from osif.data.utils.protein.vocabs import (
    ATOM_NAMES_VOCAB,
    DUMMY_TOKEN,
    RESIDUE_TO_INDEX,
    RESIDUES_VOCAB,
    AtomPropertyName,
)

EDGE_TYPE_TO_INDEX = {
    "virtual-virtual": 0,  # do not change this order
    "virtual-real": 1,
    "real-virtual": 2,
    "inter-chain": 3,  # real-real
    "inter-residue": 4,  # real-real, intra-chain
    "intra-residue": 5,  # real-real, intra-chain
}


class OSIFModel(nn.Module):
    """Torch OSIF model."""

    def __init__(
        self,
        node_hidden: int = 384,
        edge_hidden: int = 96,
        num_layers: int = 12,
        heads: int = 32,
        edge_length: float = 4.1,
        num_gaussians: int = 50,
        dropout: float = 0.15,
    ):
        """Initialize the model."""
        super().__init__()

        self.residue_type_emb = nn.Embedding(len(RESIDUES_VOCAB), node_hidden)
        self.atom_type_emb = nn.Embedding(len(ATOM_NAMES_VOCAB), node_hidden)

        self.node_emb_mlp = nn.Sequential(
            nn.Linear(2 * node_hidden, 4 * node_hidden),
            nn.LeakyReLU(),
            nn.Linear(4 * node_hidden, node_hidden),
        )

        self.edge_type_emb = nn.Embedding(len(EDGE_TYPE_TO_INDEX), edge_hidden)

        self.edge_len_emb = nn.Sequential(
            GaussianSmearing(0.0, edge_length, num_gaussians),
            nn.Linear(num_gaussians, 2 * edge_hidden),
            nn.LeakyReLU(),
            nn.Linear(2 * edge_hidden, edge_hidden),
        )
        self.edge_emb_mlp = nn.Sequential(
            nn.Linear(2 * edge_hidden, 4 * edge_hidden),
            nn.LeakyReLU(),
            nn.Linear(4 * edge_hidden, edge_hidden),
        )

        self.convs = nn.ModuleList(
            [
                GATv2Conv(
                    in_channels=node_hidden,
                    out_channels=node_hidden // heads,
                    heads=heads,
                    dropout=dropout,
                    edge_dim=edge_hidden,
                )
                for _ in range(num_layers)
            ]
        )
        self.mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LeakyReLU(),
                    nn.BatchNorm1d(node_hidden),
                    nn.Linear(node_hidden, 2 * node_hidden),
                    nn.LeakyReLU(),
                    nn.Linear(2 * node_hidden, node_hidden),
                    nn.LeakyReLU(),
                )
                for _ in range(num_layers)
            ]
        )

        self.head = nn.Sequential(
            nn.Linear(node_hidden, node_hidden),
            nn.LeakyReLU(),
            nn.Linear(node_hidden, len(RESIDUES_VOCAB)),
        )

    def _embed_edge_types(self, batch: Batch) -> torch.Tensor:
        edge_types = torch.zeros(batch.edge_index.shape[1], device=batch.edge_index.device, dtype=torch.long)
        virtual_mask_from = (
            batch.labels[batch.edge_index[0], AtomPropertyName.residue_type] == RESIDUE_TO_INDEX[DUMMY_TOKEN]
        )
        virtual_mask_to = (
            batch.labels[batch.edge_index[1], AtomPropertyName.residue_type] == RESIDUE_TO_INDEX[DUMMY_TOKEN]
        )
        edge_types[virtual_mask_from & (~virtual_mask_to)] = EDGE_TYPE_TO_INDEX["virtual-real"]
        edge_types[(~virtual_mask_from) & virtual_mask_to] = EDGE_TYPE_TO_INDEX["real-virtual"]

        chid_from = batch.labels[batch.edge_index[0], AtomPropertyName.chain]
        chid_to = batch.labels[batch.edge_index[1], AtomPropertyName.chain]

        edge_types[(~virtual_mask_from) & (~virtual_mask_to) & (chid_from != chid_to)] = EDGE_TYPE_TO_INDEX[
            "inter-chain"
        ]

        residx_from = batch.labels[batch.edge_index[0], AtomPropertyName.resindex]
        residx_to = batch.labels[batch.edge_index[1], AtomPropertyName.resindex]
        edge_types[(~virtual_mask_from) & (~virtual_mask_to) & (chid_from == chid_to) & (residx_from != residx_to)] = (
            EDGE_TYPE_TO_INDEX["inter-residue"]
        )
        edge_types[(~virtual_mask_from) & (~virtual_mask_to) & (chid_from == chid_to) & (residx_from == residx_to)] = (
            EDGE_TYPE_TO_INDEX["intra-residue"]
        )
        return self.edge_type_emb(edge_types)

    def _embed_edges(self, batch: Batch) -> torch.Tensor:
        dists = torch.norm(batch.pos[batch.edge_index[0]] - batch.pos[batch.edge_index[1]], dim=-1)
        edge_len_emb = self.edge_len_emb(dists)

        edge_type_emb = self._embed_edge_types(batch)

        return self.edge_emb_mlp(torch.cat([edge_len_emb, edge_type_emb], dim=-1))

    def _embed_nodes(self, batch: Batch) -> torch.Tensor:
        node_emb = torch.cat(
            [
                self.atom_type_emb(batch.labels[:, AtomPropertyName.atom_type]),
                self.residue_type_emb(batch.labels[:, AtomPropertyName.residue_type]),
            ],
            dim=-1,
        )
        return self.node_emb_mlp(node_emb)

    def forward(self, batch: Batch) -> torch.Tensor:
        """Forward pass of the model."""
        node_emb = self._embed_nodes(batch)

        edge_emb = self._embed_edges(batch)

        for conv, mlp in zip(self.convs, self.mlps, strict=False):
            new_node_emb = conv(x=node_emb, edge_index=batch.edge_index, edge_attr=edge_emb)
            node_emb = mlp(new_node_emb) + node_emb

        return self.head(node_emb[batch.query_mask])
