import pickle

import numpy as np
import pytest
import torch
from torch_geometric.data import HeteroData

from anemoi.graphs.edges import CutOffEdges
from anemoi.graphs.nodes import HEALPixNodes
from anemoi.models.distributed.shapes import GraphShardInfo
from anemoi.models.layers.processor import TransformerProcessor
from anemoi.models.layers.utils import load_layer_kernels

from nese_training.radius_attention import MaskedAttentionWrapper
from nese_training.radius_attention import RadiusTransformerProcessor
from nese_training.radius_attention import adjacency_mask

EARTH_RADIUS_KM = 6371.0
RADIUS_KM = 4000.0
GLOBAL_RADIUS_KM = 13000.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def hidden_graph(radius_km: float, nest_ordering: bool = True) -> HeteroData:
    """Graph on ``DEVICE``, since anemoi's builders pick the device themselves."""
    graph = HEALPixNodes(resolution=2, name="hidden", nest_ordering=nest_ordering).update_graph(HeteroData())
    edges = CutOffEdges("hidden", "hidden", cutoff_distance_km=radius_km, max_num_neighbours=1024)
    return edges.update_graph(graph).to(DEVICE)


def unit_vectors(graph: HeteroData) -> np.ndarray:
    lat, lon = graph["hidden"].x.cpu().numpy().T
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)


def processor_config(num_layers: int = 2) -> dict:
    return dict(
        num_layers=num_layers,
        num_channels=32,
        num_chunks=1,
        num_heads=4,
        mlp_hidden_ratio=2,
        dropout_p=0.0,
        attention_implementation="scaled_dot_product_attention",
        layer_kernels=load_layer_kernels(instance=False),
    )


def batched_edge_index(edge_index: torch.Tensor, num_nodes: int, batch_size: int) -> torch.Tensor:
    return torch.cat([edge_index + i * num_nodes for i in range(batch_size)], dim=1)


@pytest.mark.parametrize("nest_ordering", [True, False])
def test_mask_is_symmetric_disk_with_self_attention(nest_ordering):
    graph = hidden_graph(RADIUS_KM, nest_ordering)
    mask = adjacency_mask(graph["hidden", "to", "hidden"].edge_index, graph["hidden"].num_nodes).cpu().numpy()

    xyz = unit_vectors(graph)
    chord = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
    expected = chord <= RADIUS_KM / EARTH_RADIUS_KM

    np.testing.assert_array_equal(mask, expected)
    np.testing.assert_array_equal(mask, mask.T)
    assert mask.diagonal().all()


def test_batched_edges_fold_onto_same_mask():
    graph = hidden_graph(RADIUS_KM)
    num_nodes = graph["hidden"].num_nodes
    edge_index = graph["hidden", "to", "hidden"].edge_index
    torch.testing.assert_close(
        adjacency_mask(batched_edge_index(edge_index, num_nodes, 3), num_nodes),
        adjacency_mask(edge_index, num_nodes),
    )


def test_masked_sdpa_matches_naive_softmax():
    torch.manual_seed(0)
    query, key, value = torch.randn(3, 2, 4, 50, 8, device=DEVICE).unbind(0)
    mask = torch.rand(50, 50, device=DEVICE) < 0.2
    mask |= torch.eye(50, dtype=torch.bool, device=DEVICE)

    attention = MaskedAttentionWrapper("scaled_dot_product_attention")
    attention.set_mask(mask)

    logits = query @ key.transpose(-1, -2) / 8**0.5
    expected = logits.masked_fill(~mask, float("-inf")).softmax(-1) @ value
    torch.testing.assert_close(attention(query, key, value, batch_size=2), expected)


@pytest.mark.parametrize(
    "block_size, compile_mode",
    [(128, None), ((128, 64), None), (64, "max-autotune-no-cudagraphs")],
)
def test_flex_matches_sdpa(block_size, compile_mode):
    torch.manual_seed(0)
    query, key, value = torch.randn(3, 1, 4, 192, 16, device=DEVICE).unbind(0)
    graph = hidden_graph(RADIUS_KM)
    mask = adjacency_mask(graph["hidden", "to", "hidden"].edge_index, graph["hidden"].num_nodes)

    sdpa = MaskedAttentionWrapper("scaled_dot_product_attention")
    flex = MaskedAttentionWrapper("flex_attention", block_size, compile_mode)
    sdpa.set_mask(mask)
    flex.set_mask(mask)
    assert flex.block_mask.BLOCK_SIZE == (block_size if isinstance(block_size, tuple) else (block_size, block_size))
    tolerance = 5e-3 if DEVICE.type == "cuda" else 1e-5  # the flex triton kernel matmuls in tf32
    torch.testing.assert_close(flex(query, key, value, 1), sdpa(query, key, value, 1), atol=tolerance, rtol=tolerance)


def test_global_radius_matches_full_attention_transformer():
    torch.manual_seed(0)
    graph = hidden_graph(GLOBAL_RADIUS_KM)
    num_nodes, batch_size = graph["hidden"].num_nodes, 2
    edge_index = batched_edge_index(graph["hidden", "to", "hidden"].edge_index, num_nodes, batch_size)

    reference = TransformerProcessor(**processor_config()).to(DEVICE)
    radius = RadiusTransformerProcessor(**processor_config()).to(DEVICE)
    radius.load_state_dict(reference.state_dict())

    x = torch.randn(batch_size * num_nodes, 32, device=DEVICE)
    shard_info = GraphShardInfo(nodes=[num_nodes])
    torch.testing.assert_close(
        radius(x, batch_size, shard_info, edge_index=edge_index),
        reference(x, batch_size, shard_info),
    )


def test_single_layer_only_sees_nodes_inside_radius():
    torch.manual_seed(0)
    graph = hidden_graph(RADIUS_KM)
    num_nodes = graph["hidden"].num_nodes
    edge_index = graph["hidden", "to", "hidden"].edge_index
    mask = adjacency_mask(edge_index, num_nodes)

    processor = RadiusTransformerProcessor(**processor_config(num_layers=1)).to(DEVICE).eval()
    shard_info = GraphShardInfo(nodes=[num_nodes])
    x = torch.randn(num_nodes, 32, device=DEVICE, requires_grad=True)

    query_node = 0
    output = processor(x, 1, shard_info, edge_index=edge_index)
    (gradient,) = torch.autograd.grad(output[query_node].sum(), x)
    influenced = gradient.abs().sum(-1) > 0

    torch.testing.assert_close(influenced, mask[query_node])
    assert 1 < influenced.sum() < num_nodes


def test_requires_hidden_edges():
    processor = RadiusTransformerProcessor(**processor_config())
    with pytest.raises(AssertionError, match="hidden, to, hidden"):
        processor(torch.randn(10, 32, device=DEVICE), 1, GraphShardInfo(nodes=[10]))


def test_rejects_unknown_attention_implementation():
    with pytest.raises(ValueError, match="attention_implementation"):
        RadiusTransformerProcessor(**{**processor_config(), "attention_implementation": "flash_attention"})


def test_rejects_dropout_with_flex_attention():
    config = {**processor_config(), "attention_implementation": "flex_attention", "dropout_p": 0.1}
    with pytest.raises(ValueError, match="dropout"):
        RadiusTransformerProcessor(**config)


def test_compile_mode_is_passed_to_masked_attention():
    config = {**processor_config(), "attention_implementation": "flex_attention"}
    processor = RadiusTransformerProcessor(**config, compile_mode="max-autotune-no-cudagraphs")
    assert processor.masked_attention.compile_mode == "max-autotune-no-cudagraphs"


def test_dropout_is_passed_to_masked_sdpa():
    torch.manual_seed(0)
    processor = RadiusTransformerProcessor(**{**processor_config(num_layers=1), "dropout_p": 0.5}).to(DEVICE).train()
    graph = hidden_graph(RADIUS_KM)
    num_nodes = graph["hidden"].num_nodes
    edge_index = graph["hidden", "to", "hidden"].edge_index

    x = torch.randn(num_nodes, 32, device=DEVICE)
    shard_info = GraphShardInfo(nodes=[num_nodes])
    assert not torch.allclose(
        processor(x, 1, shard_info, edge_index=edge_index),
        processor(x, 1, shard_info, edge_index=edge_index),
    )


def test_flex_attention_processor_survives_pickling():
    """anemoi pickles the model at train start, and flex attention state is unpicklable."""
    torch.manual_seed(0)
    graph = hidden_graph(RADIUS_KM)
    num_nodes = graph["hidden"].num_nodes
    edge_index = graph["hidden", "to", "hidden"].edge_index

    # 64 channels over 4 heads: the flex attention backward needs a head dimension of at least 16.
    config = {**processor_config(num_layers=1), "attention_implementation": "flex_attention", "num_channels": 64}
    processor = RadiusTransformerProcessor(**config).to(DEVICE).eval()
    x = torch.randn(num_nodes, 64, device=DEVICE)
    shard_info = GraphShardInfo(nodes=[num_nodes])
    expected = processor(x, 1, shard_info, edge_index=edge_index)

    reloaded = pickle.loads(pickle.dumps(processor))
    assert reloaded.masked_attention.mask is None
    torch.testing.assert_close(reloaded(x, 1, shard_info, edge_index=edge_index), expected)
