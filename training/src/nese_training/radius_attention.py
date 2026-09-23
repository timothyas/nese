"""Transformer processor whose attention stencil is the hidden-to-hidden graph.

The mask is built from the ``(hidden, to, hidden)`` edge set that anemoi's
``StaticGraphProvider`` already passes to ``processor.forward``, so a
``CutOffEdges`` set gives a great-circle disk per layer (GenCast-style sparse
attention, but radius-based rather than k-hop).  Node ordering only matters for
the block sparsity seen by the ``flex_attention`` implementation, not for the stencil.

Only single-process (unsharded) edges are supported for now: with a model comm
group anemoi shards ``edge_index`` per rank, so the full mask would need to be
gathered first.
"""

from typing import Literal
from typing import get_args
from typing import Optional
from typing import Union

import torch
from torch import Tensor
from torch import nn
from torch.nn.functional import scaled_dot_product_attention

from anemoi.models.distributed.utils import model_is_distributed
from anemoi.models.layers.processor import TransformerProcessor


def adjacency_mask(edge_index: Tensor, num_nodes: int) -> Tensor:
    """Dense ``[receiver, sender]`` boolean mask with self attention included.

    Batch-expanded edge indices (offset by multiples of ``num_nodes``) fold back
    onto the same mask.
    """
    senders, receivers = edge_index % num_nodes
    mask = torch.eye(num_nodes, dtype=torch.bool, device=edge_index.device)
    mask[receivers, senders] = True
    return mask


AttentionImplementation = Literal["scaled_dot_product_attention", "flex_attention"]
BlockSize = Union[int, tuple[int, int]]


class MaskedAttentionWrapper(nn.Module):
    """Drop-in for anemoi's attention wrappers that applies a fixed node mask."""

    def __init__(
        self,
        attention_implementation: AttentionImplementation = "scaled_dot_product_attention",
        block_size: BlockSize = 128,
        compile_mode: Optional[str] = None,
    ):
        super().__init__()
        self.attention_implementation = attention_implementation
        self.block_size = block_size
        self.compile_mode = compile_mode
        self.mask: Optional[Tensor] = None
        self.block_mask = None
        self.flex_attention = None

    def set_mask(self, mask: Tensor) -> None:
        self.mask = mask
        self.block_mask = None
        if self.attention_implementation == "flex_attention":
            from torch.nn.attention.flex_attention import create_block_mask
            from torch.nn.attention.flex_attention import flex_attention

            self.flex_attention = (
                torch.compile(flex_attention, mode=self.compile_mode) if mask.is_cuda else flex_attention
            )

            def mask_mod(batch, head, q_idx, kv_idx):
                return mask[q_idx, kv_idx]

            num_nodes = mask.shape[0]
            self.block_mask = create_block_mask(
                mask_mod, None, None, num_nodes, num_nodes, device=mask.device, BLOCK_SIZE=self.block_size
            )

    def __getstate__(self) -> dict:
        """The mask, block mask, and compiled kernel are rebuilt on the next forward pass.

        Dropping them keeps checkpoints free of the dense mask and of objects that
        cannot be pickled at all, like the ``mask_mod`` closure.
        """
        return {**super().__getstate__(), "mask": None, "block_mask": None, "flex_attention": None}

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        batch_size: int,
        causal: bool = False,
        window_size: Optional[int] = None,
        dropout_p: float = 0.0,
        softcap: Optional[float] = None,
        alibi_slopes: Optional[Tensor] = None,
    ) -> Tensor:
        assert self.mask is not None, "set_mask must be called before the first forward pass"
        assert not causal and window_size is None, "the node mask replaces causal and sliding-window masking"
        assert not softcap and alibi_slopes is None, "softcap and alibi slopes are not supported"
        if self.attention_implementation == "flex_attention":
            assert dropout_p == 0.0, "flex_attention does not support attention dropout"
            return self.flex_attention(query, key, value, block_mask=self.block_mask)
        return scaled_dot_product_attention(query, key, value, attn_mask=self.mask, dropout_p=dropout_p)


class RadiusTransformerProcessor(TransformerProcessor):
    """``TransformerProcessor`` attending only along the hidden-to-hidden edges."""

    def __init__(
        self,
        *,
        attention_implementation: AttentionImplementation = "scaled_dot_product_attention",
        block_size: BlockSize = 128,
        compile_mode: Optional[str] = None,
        dropout_p: float = 0.0,
        **kwargs,
    ) -> None:
        supported = get_args(AttentionImplementation)
        if attention_implementation not in supported:
            raise ValueError(f"attention_implementation must be one of {supported}, got {attention_implementation!r}")
        if attention_implementation == "flex_attention" and dropout_p > 0.0:
            raise ValueError("flex_attention does not support attention dropout, set dropout_p to 0")
        kwargs["window_size"] = None
        super().__init__(attention_implementation="scaled_dot_product_attention", dropout_p=dropout_p, **kwargs)
        self.masked_attention = MaskedAttentionWrapper(attention_implementation, block_size, compile_mode)
        for block in self.proc:
            block.attention.attention = self.masked_attention

    def forward(
        self,
        x: Tensor,
        batch_size: int,
        shard_info,
        edge_attr: Optional[Tensor] = None,
        edge_index: Optional[Tensor] = None,
        model_comm_group=None,
        **kwargs,
    ) -> Tensor:
        assert edge_index is not None, "RadiusTransformerProcessor needs a (hidden, to, hidden) edge set in the graph"
        assert not model_is_distributed(model_comm_group), "sharded processor edges are not supported yet"
        num_nodes = x.shape[0] // batch_size
        mask = self.masked_attention.mask
        if mask is None or mask.shape[0] != num_nodes or mask.device != x.device:
            self.masked_attention.set_mask(adjacency_mask(edge_index.to(x.device), num_nodes))
        return super().forward(x, batch_size, shard_info, model_comm_group=model_comm_group, **kwargs)
