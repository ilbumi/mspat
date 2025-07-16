"""En Transformer adopted from Lucidrain's repository."""

from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as func
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from einx import get_at
from taylor_series_linear_attention import TaylorSeriesLinearAttn
from torch import einsum, nn
from torch.utils.checkpoint import checkpoint_sequential

# helper functions


def exists(val: Any) -> bool:
    """Check if a value exists (is not None)."""
    return val is not None


def max_neg_value(t) -> float:
    """Return the maximum negative value for a given tensor type."""
    return -torch.finfo(t.dtype).max


def default(val: Any, d: Any) -> Any:
    """Return the default value if the given value does not exist."""
    return val if exists(val) else d


def l2norm(t: torch.Tensor) -> torch.Tensor:
    """Apply L2 normalization to the last dimension of a tensor."""
    return func.normalize(t, dim=-1)


def small_init_(t: nn.Linear) -> None:
    """Initialize a linear layer with small weights and zero bias."""
    nn.init.normal_(t.weight, std=0.02)
    nn.init.zeros_(t.bias)


# dynamic positional bias


class DynamicPositionBias(nn.Module):
    def __init__(
        self, dim: int, *, heads: int, depth: int, dim_head: int, input_dim: int = 1, norm: bool = True
    ) -> None:
        """Initialize the DynamicPositionBias module.

        Args:
            dim (int): dimensionality of the input features.
            heads (int): number of attention heads.
            depth (int): depth of the MLP.
            dim_head (int): dimensionality of each attention head.
            input_dim (int, optional): dimensionality of input. Defaults to 1.
            norm (bool, optional): wether to perform validation. Defaults to True.

        """
        super().__init__()
        assert depth >= 1, "depth for dynamic position bias MLP must be greater or equal to 1"  # noqa: S101
        self.mlp = nn.ModuleList([])

        self.mlp.append(
            nn.Sequential(nn.Linear(input_dim, dim), nn.LayerNorm(dim) if norm else nn.Identity(), nn.SiLU())
        )

        for _ in range(depth - 1):
            self.mlp.append(nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim) if norm else nn.Identity(), nn.SiLU()))

        self.heads = heads
        self.qk_pos_head = nn.Linear(dim, heads)
        self.value_pos_head = nn.Linear(dim, dim_head * heads)

    def forward(self, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for the DynamicPositionBias module."""
        for layer in self.mlp:
            pos = layer(pos)

        qk_pos = self.qk_pos_head(pos)
        value_pos = self.value_pos_head(pos)

        qk_pos = rearrange(qk_pos, "b 1 i j h -> b h i j")
        value_pos = rearrange(value_pos, "b 1 i j (h d) -> b h i j d", h=self.heads)
        return qk_pos, value_pos


# classes

# this follows the same strategy for normalization as done in SE3 Transformers
# https://github.com/lucidrains/se3-transformer-pytorch/blob/main/se3_transformer_pytorch/se3_transformer_pytorch.py#L95


class LayerNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        """Initialize the LayerNorm module."""
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the LayerNorm module."""
        return func.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


class CoorsNorm(nn.Module):
    def __init__(self, eps: float = 1e-8, scale_init: float = 1.0) -> None:
        """Initialize the CoorsNorm module."""
        super().__init__()
        self.eps = eps
        scale = torch.zeros(1).fill_(scale_init)
        self.scale = nn.Parameter(scale)

    def forward(self, coors: torch.Tensor) -> torch.Tensor:
        """Forward pass for the CoorsNorm module."""
        norm = coors.norm(dim=-1, keepdim=True)
        normed_coors = coors / norm.clamp(min=self.eps)
        return normed_coors * self.scale


class Residual(nn.Module):
    def __init__(self, fn: Callable[..., tuple[torch.Tensor, torch.Tensor]]) -> None:
        """Initialize the Residual module."""
        super().__init__()
        self.fn = fn

    def forward(self, feats: torch.Tensor, coors: torch.Tensor, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for the Residual module."""
        feats_out, coors_delta = self.fn(feats, coors, **kwargs)
        return feats + feats_out, coors + coors_delta


class GEGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the GEGLU module."""
        x, gates = x.chunk(2, dim=-1)
        return x * func.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, *, dim: int, mult: int = 4, dropout: float = 0.0) -> None:
        """Initialize the FeedForward module."""
        super().__init__()
        inner_dim = int(dim * mult * 2 / 3)

        self.net = nn.Sequential(
            LayerNorm(dim),
            nn.Linear(dim, inner_dim * 2, bias=False),
            GEGLU(),
            LayerNorm(inner_dim),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim, bias=False),
        )

    def forward(self, feats: torch.Tensor, coors: Any) -> tuple[torch.Tensor, Any]:  # noqa: ARG002
        """Forward pass for the FeedForward module."""
        return self.net(feats), 0


class EquivariantAttention(nn.Module):
    def __init__(  # noqa: PLR0915
        self,
        *,
        dim,
        dim_head: int = 64,
        heads: int = 4,
        edge_dim: int = 0,
        coors_hidden_dim: int = 16,
        neighbors: int = 0,
        only_sparse_neighbors: bool = False,
        valid_neighbor_radius: float = float("inf"),
        init_eps: float = 1e-3,
        rel_pos_emb=None,
        edge_mlp_mult: int = 2,
        norm_rel_coors: bool = True,
        norm_coors_scale_init: float = 1.0,
        use_cross_product: bool = False,
        talking_heads: bool = False,
        dropout: float = 0.0,
        num_global_linear_attn_heads: int = 0,
        linear_attn_dim_head: int = 8,
        gate_outputs: bool = True,
        gate_init_bias: float = 10.0,
    ) -> None:
        """Initialize the EquivariantAttention module."""
        super().__init__()
        neighbors = default(neighbors, 0)
        neighbors = max(neighbors, 0)

        self.scale = dim_head**-0.5
        self.norm = LayerNorm(dim)

        self.neighbors = neighbors
        self.only_sparse_neighbors = only_sparse_neighbors
        self.valid_neighbor_radius = valid_neighbor_radius

        attn_inner_dim = heads * dim_head
        self.heads = heads

        self.has_linear_attn = num_global_linear_attn_heads > 0

        linear_attn_dim_hidden = 0
        if self.has_linear_attn:
            self.linear_attn = TaylorSeriesLinearAttn(
                dim=dim,
                dim_head=linear_attn_dim_head,
                heads=num_global_linear_attn_heads,
                gate_value_heads=True,
                combine_heads=False,
            )

            linear_attn_dim_hidden = self.linear_attn.dim_hidden

        self.to_qkv = nn.Linear(dim, attn_inner_dim * 3, bias=False)
        self.to_out = nn.Linear(attn_inner_dim + linear_attn_dim_hidden, dim)

        self.gate_outputs = gate_outputs
        if gate_outputs:
            gate_linear = nn.Linear(dim, 2 * heads)
            nn.init.zeros_(gate_linear.weight)
            nn.init.constant_(gate_linear.bias, gate_init_bias)

            self.to_output_gates = nn.Sequential(
                gate_linear, nn.Sigmoid(), Rearrange("b n (l h) -> l b h n 1", h=heads)
            )

        self.talking_heads = nn.Conv2d(heads, heads, 1, bias=False) if talking_heads else None

        self.edge_mlp = None
        has_edges = edge_dim > 0

        if has_edges:
            edge_input_dim = heads + edge_dim
            edge_hidden = edge_input_dim * edge_mlp_mult

            self.edge_mlp = nn.Sequential(
                nn.Linear(edge_input_dim, edge_hidden, bias=False), nn.GELU(), nn.Linear(edge_hidden, heads, bias=False)
            )

            self.coors_mlp = nn.Sequential(nn.GELU(), nn.Linear(heads, heads, bias=False))
        else:
            self.coors_mlp = nn.Sequential(
                nn.Linear(heads, coors_hidden_dim, bias=False),
                nn.GELU(),
                nn.Linear(coors_hidden_dim, heads, bias=False),
            )

        self.coors_gate = nn.Linear(heads, heads)
        small_init_(self.coors_gate)

        self.use_cross_product = use_cross_product
        if use_cross_product:
            self.cross_coors_mlp = nn.Sequential(
                nn.Linear(heads, coors_hidden_dim, bias=False),
                nn.GELU(),
                nn.Linear(coors_hidden_dim, heads * 2, bias=False),
            )

            self.cross_coors_gate_i = nn.Linear(heads, heads)
            self.cross_coors_gate_j = nn.Linear(heads, heads)

            small_init_(self.cross_coors_gate_i)
            small_init_(self.cross_coors_gate_j)

        self.norm_rel_coors = CoorsNorm(scale_init=norm_coors_scale_init) if norm_rel_coors else nn.Identity()

        num_coors_combine_heads = (2 if use_cross_product else 1) * heads
        self.coors_combine = nn.Parameter(torch.randn(num_coors_combine_heads))

        # positional embedding
        # for both along the sequence (if specified by rel_pos_emb)
        # and the relative distance between each residue / atom

        self.rel_pos_emb = rel_pos_emb

        self.dynamic_pos_bias_mlp = DynamicPositionBias(
            dim=dim // 2, heads=heads, dim_head=dim_head, depth=3, input_dim=(2 if rel_pos_emb else 1)
        )

        # dropouts

        self.node_dropout = nn.Dropout(dropout)
        self.coor_dropout = nn.Dropout(dropout)

        # init

        self.init_eps = init_eps
        self.apply(self.init_)

    def init_(self, module: nn.Module) -> None:
        """Initialize the module weights."""
        if type(module) in {nn.Linear}:
            nn.init.normal_(module.weight, std=self.init_eps)

    def forward(  # noqa: C901, PLR0912, PLR0915
        self,
        feats: torch.Tensor,
        coors: torch.Tensor,
        edges: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        adj_mat: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for the EquivariantAttention module."""
        b, n, _, h, num_nn, only_sparse_neighbors, valid_neighbor_radius, device = (
            *feats.shape,
            self.heads,
            self.neighbors,
            self.only_sparse_neighbors,
            self.valid_neighbor_radius,
            feats.device,
        )

        _mask = mask
        feats = self.norm(feats)

        assert not (only_sparse_neighbors and not exists(adj_mat)), (  # noqa: S101
            "adjacency matrix must be passed in if only_sparse_neighbors is turned on"
        )

        rel_coors = rearrange(coors, "b i d -> b i 1 d") - rearrange(coors, "b j d -> b 1 j d")
        rel_dist = rel_coors.norm(p=2, dim=-1)

        # calculate neighborhood indices

        nbhd_indices = None
        nbhd_masks = None
        nbhd_ranking = rel_dist.clone()

        if exists(adj_mat):
            if len(adj_mat.shape) == 2:  # noqa: PLR2004
                adj_mat = repeat(adj_mat, "i j -> b i j", b=b)

            self_mask = torch.eye(n, device=device).bool()
            self_mask = rearrange(self_mask, "i j -> 1 i j")
            adj_mat.masked_fill_(self_mask, False)  # noqa: FBT003

            max_adj_neighbors = adj_mat.long().sum(dim=-1).max().item() + 1

            num_nn = max_adj_neighbors if only_sparse_neighbors else (num_nn + max_adj_neighbors)
            valid_neighbor_radius = 0 if only_sparse_neighbors else valid_neighbor_radius

            nbhd_ranking = nbhd_ranking.masked_fill(self_mask, -1.0)
            nbhd_ranking = nbhd_ranking.masked_fill(adj_mat, 0.0)

        if 0 < num_nn < n:
            # make sure padding does not end up becoming neighbors
            if exists(mask):
                ranking_mask = mask[:, :, None] * mask[:, None, :]
                nbhd_ranking = nbhd_ranking.masked_fill(~ranking_mask, 1e5)

            nbhd_values, nbhd_indices = nbhd_ranking.topk(num_nn, dim=-1, largest=False)
            nbhd_masks = nbhd_values <= valid_neighbor_radius

        # derive queries keys and values

        q, k, v = self.to_qkv(feats).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=h) for t in (q, k, v))

        # to give network ability to attend to nothing

        if self.gate_outputs:
            out_gate, rel_out_gate = self.to_output_gates(feats)

        # calculate nearest neighbors

        i = j = n

        if exists(nbhd_indices):
            i, j = nbhd_indices.shape[-2:]
            k = get_at("b h [j] d, b i k -> b h i k d", k, nbhd_indices)
            v = get_at("b h [j] d, b i k -> b h i k d", v, nbhd_indices)
            rel_dist = get_at("b i [j], b i k -> b i k", rel_dist, nbhd_indices)
            rel_coors = get_at("b i [j] c, b i k -> b i k c", rel_coors, nbhd_indices)
        else:
            k = repeat(k, "b h j d -> b h n j d", n=n)
            v = repeat(v, "b h j d -> b h n j d", n=n)

        # prepare mask

        if exists(mask):
            q_mask = rearrange(mask, "b i -> b 1 i 1")

            if exists(nbhd_indices):
                k_mask = get_at("b [j], b i k -> b 1 i k", mask, nbhd_indices)

            mask = q_mask * k_mask

            if exists(nbhd_masks):
                mask &= rearrange(nbhd_masks, "b i j -> b 1 i j")

        # generate and apply rotary embeddings

        rel_dist = -(rel_dist**2)
        rel_dist = rearrange(rel_dist, "b i j -> b 1 i j 1")

        if self.rel_pos_emb:
            seq = torch.arange(n, device=device, dtype=q.dtype)
            seq_target_pos = nbhd_indices if exists(nbhd_indices) else rearrange(seq, "j -> 1 1 j")
            seq_rel_dist = rearrange(seq, "i -> 1 i 1") - seq_target_pos
            seq_rel_dist = repeat(seq_rel_dist, "b i j -> b 1 i j 1", b=b)
            rel_dist = torch.cat((rel_dist, seq_rel_dist), dim=-1)

        qk_pos, value_pos = self.dynamic_pos_bias_mlp(rel_dist)

        # calculate inner product for queries and keys

        q = repeat(q, "b h i d -> b h i j d", j=k.shape[-2])

        # l2 distance
        # -cdist(q, k).pow(2)  # noqa: ERA001

        qk = -((q - k) ** 2).sum(dim=-1)

        qk = qk * self.scale

        # add relative positions to qk as well as values

        qk = qk + qk_pos

        v = v + value_pos

        # add edge information and pass through edges MLP if needed

        if exists(edges):
            if exists(nbhd_indices):
                edges = get_at("b i [j] d, b i k -> b i k d", edges, nbhd_indices)

            qk = rearrange(qk, "b h i j -> b i j h")
            qk = torch.cat((qk, edges), dim=-1)
            qk = self.edge_mlp(qk)
            qk = rearrange(qk, "b i j h -> b h i j")

        # coordinate MLP and calculate coordinate updates

        coors_mlp_input = rearrange(qk, "b h i j -> b i j h")
        coor_weights = self.coors_mlp(coors_mlp_input)

        if exists(mask):
            mask_value = max_neg_value(coor_weights)
            coor_mask = repeat(mask, "b 1 i j -> b i j 1")
            coor_weights.masked_fill_(~coor_mask, mask_value)

        coor_attn = coor_weights.softmax(dim=-2)
        coor_attn = self.coor_dropout(coor_attn)

        rel_coors_sign = self.coors_gate(coors_mlp_input)
        rel_coors_sign = rearrange(rel_coors_sign, "b i j h -> b i j 1 h")

        if self.use_cross_product:
            rel_coors_i = repeat(rel_coors, "b n i c -> b n (i j) c", j=j)
            rel_coors_j = repeat(rel_coors, "b n j c -> b n (i j) c", i=j)

            cross_coors = torch.cross(rel_coors_i, rel_coors_j, dim=-1)

            cross_coors = self.norm_rel_coors(cross_coors)
            cross_coors = repeat(cross_coors, "b i j c -> b i j c h", h=h)

            cross_coors_sign_i = self.cross_coors_gate_i(coors_mlp_input)
            cross_coors_sign_j = self.cross_coors_gate_j(coors_mlp_input)

            cross_coors_sign = rearrange(cross_coors_sign_i, "b n i h -> b n i 1 h") * rearrange(
                cross_coors_sign_j, "b n j h -> b n 1 j h"
            )
            cross_coors_sign = rearrange(cross_coors_sign, "b n i j h -> b n (i j) 1 h")

            cross_coors = cross_coors * cross_coors_sign

        rel_coors = self.norm_rel_coors(rel_coors)
        rel_coors = repeat(rel_coors, "b i j c -> b i j c h", h=h)

        rel_coors = rel_coors * rel_coors_sign

        # cross product

        if self.use_cross_product:
            cross_weights = self.cross_coors_mlp(coors_mlp_input)

            cross_weights = rearrange(cross_weights, "b i j (h n) -> b i j h n", n=2)
            cross_weights_i, cross_weights_j = cross_weights.unbind(dim=-1)

            cross_weights = rearrange(cross_weights_i, "b n i h -> b n i 1 h") + rearrange(
                cross_weights_j, "b n j h -> b n 1 j h"
            )

            if exists(mask):
                cross_mask = coor_mask[:, :, :, None, :] & coor_mask[:, :, None, :, :]
                cross_weights = cross_weights.masked_fill(~cross_mask, mask_value)

            cross_weights = rearrange(cross_weights, "b n i j h -> b n (i j) h")
            cross_attn = cross_weights.softmax(dim=-2)

        # aggregate and combine heads for coordinate updates

        rel_out = einsum("b i j h, b i j c h -> b h i c", coor_attn, rel_coors)

        if self.gate_outputs:
            rel_out = rel_out * rel_out_gate

        if self.use_cross_product:
            cross_out = einsum("b i j h, b i j c h -> b h i c", cross_attn, cross_coors)
            rel_out = torch.cat((rel_out, cross_out), dim=1)

        coors_out = einsum("b h n c, h -> b n c", rel_out, self.coors_combine)

        # derive attention

        sim = qk.clone()

        if exists(mask):
            mask_value = max_neg_value(sim)
            sim.masked_fill_(~mask, mask_value)

        attn = sim.softmax(dim=-1)
        attn = self.node_dropout(attn)

        if exists(self.talking_heads):
            attn = self.talking_heads(attn)

        # weighted sum of values and combine heads

        out = einsum("b h i j, b h i j d -> b h i d", attn, v)

        if self.gate_outputs:
            out = out * out_gate

        out = rearrange(out, "b h n d -> b n (h d)")

        # linear attention

        if self.has_linear_attn:
            lin_out = self.linear_attn(feats, mask=_mask)
            out = torch.cat((out, lin_out), dim=-1)

        # combine heads, both local + global linear attention (if designated)

        out = self.to_out(out)

        return out, coors_out


class Block(nn.Module):
    def __init__(
        self,
        attn: Callable[..., tuple[torch.Tensor, torch.Tensor]],
        ff: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """Initialize the Block module."""
        super().__init__()
        self.attn = attn
        self.ff = ff

    def forward(
        self,
        inp: torch.Tensor,
        coor_changes: Any = None,  # noqa: ARG002
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for the Block module."""
        feats, coors, mask, edges, adj_mat = inp
        feats, coors = self.attn(feats, coors, edges=edges, mask=mask, adj_mat=adj_mat)
        feats, coors = self.ff(feats, coors)
        return (feats, coors, mask, edges, adj_mat)


class EnTransformer(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        depth: int,
        num_tokens: int | None = None,
        rel_pos_emb: bool = False,
        dim_head: int = 64,
        heads: int = 8,
        num_edge_tokens: int | None = None,
        edge_dim: int = 0,
        coors_hidden_dim: int = 16,
        neighbors: int = 0,
        only_sparse_neighbors: bool = False,
        num_adj_degrees: int | None = None,
        adj_dim: int = 0,
        valid_neighbor_radius: float = float("inf"),
        init_eps: float = 1e-3,
        norm_rel_coors: bool = True,
        norm_coors_scale_init: float = 1.0,
        use_cross_product: bool = False,
        talking_heads: bool = False,
        checkpoint: bool = False,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        num_global_linear_attn_heads: int = 0,
        gate_outputs: bool = True,
    ) -> None:
        """Initialize the EnTransformer module."""
        super().__init__()
        assert dim_head >= 32, "your dimension per head should be greater than 32 for rotary embeddings to work well"  # noqa: PLR2004, S101
        assert not (exists(num_adj_degrees) and num_adj_degrees < 1), "make sure adjacent degrees is greater than 1"  # noqa: S101

        if only_sparse_neighbors:
            num_adj_degrees = default(num_adj_degrees, 1)

        self.token_emb: nn.Embedding | None = nn.Embedding(num_tokens, dim) if exists(num_tokens) else None
        self.edge_emb: nn.Embedding | None = (
            nn.Embedding(num_edge_tokens, edge_dim) if exists(num_edge_tokens) else None
        )

        self.num_adj_degrees: int | None = num_adj_degrees
        self.adj_emb: nn.Embedding | None = (
            nn.Embedding(num_adj_degrees + 1, adj_dim) if exists(num_adj_degrees) and adj_dim > 0 else None
        )
        adj_dim = adj_dim if exists(num_adj_degrees) else 0

        self.checkpoint: bool = checkpoint
        self.layers: nn.ModuleList = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                Block(
                    Residual(
                        EquivariantAttention(
                            dim=dim,
                            dim_head=dim_head,
                            heads=heads,
                            coors_hidden_dim=coors_hidden_dim,
                            edge_dim=(edge_dim + adj_dim),
                            neighbors=neighbors,
                            only_sparse_neighbors=only_sparse_neighbors,
                            valid_neighbor_radius=valid_neighbor_radius,
                            init_eps=init_eps,
                            rel_pos_emb=rel_pos_emb,
                            norm_rel_coors=norm_rel_coors,
                            norm_coors_scale_init=norm_coors_scale_init,
                            use_cross_product=use_cross_product,
                            talking_heads=talking_heads,
                            dropout=attn_dropout,
                            num_global_linear_attn_heads=num_global_linear_attn_heads,
                            gate_outputs=gate_outputs,
                        )
                    ),
                    Residual(FeedForward(dim=dim, dropout=ff_dropout)),
                )
            )

    def forward(
        self,
        feats: torch.Tensor,
        coors: torch.Tensor,
        edges: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        adj_mat: torch.Tensor | None = None,
        return_coor_changes: bool = False,
        **kwargs,  # noqa: ARG002
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """Forward pass for the EnTransformer module."""
        b = feats.shape[0]

        if exists(self.token_emb):
            feats = self.token_emb(feats)

        if exists(self.edge_emb):
            assert exists(edges), "edges must be passed in as (batch x seq x seq) indicating edge type"  # noqa: S101
            edges = self.edge_emb(edges)

        assert not (exists(adj_mat) and (not exists(self.num_adj_degrees) or self.num_adj_degrees == 0)), (  # noqa: S101
            "num_adj_degrees must be greater than 0 if you are passing in an adjacency matrix"
        )

        if exists(self.num_adj_degrees):
            assert exists(adj_mat), "adjacency matrix must be passed in (keyword argument adj_mat)"  # noqa: S101

            if len(adj_mat.shape) == 2:  # noqa: PLR2004
                adj_mat = repeat(adj_mat.clone(), "i j -> b i j", b=b)

            adj_indices = adj_mat.clone().long()

            for ind in range(self.num_adj_degrees - 1):
                degree = ind + 2

                next_degree_adj_mat = (adj_mat.float() @ adj_mat.float()) > 0
                next_degree_mask = (next_degree_adj_mat.float() - adj_mat.float()).bool()
                adj_indices.masked_fill_(next_degree_mask, degree)
                adj_mat = next_degree_adj_mat.clone()

            if exists(self.adj_emb):
                adj_emb = self.adj_emb(adj_indices)
                edges = torch.cat((edges, adj_emb), dim=-1) if exists(edges) else adj_emb

        assert not (return_coor_changes and self.training), "you must be eval mode in order to return coordinates"  # noqa: S101

        # go through layers

        coor_changes: list[torch.Tensor] = [coors]
        inp = (feats, coors, mask, edges, adj_mat)

        # if in training mode and checkpointing is designated, use checkpointing across blocks to save memory

        if self.training and self.checkpoint:
            inp = checkpoint_sequential(self.layers, len(self.layers), inp)
        else:
            # iterate through blocks
            for layer in self.layers:
                inp = layer(inp)
                coor_changes.append(inp[1])  # append coordinates for visualization

        feats, coors, *_ = inp

        if return_coor_changes:
            return feats, coors, coor_changes

        return feats, coors
