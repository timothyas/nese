"""Time one processor forward+backward per flex_attention block size, on one GPU.

Dimensions follow training.yaml (12288 hidden nodes, 512 channels, 16 heads,
16 layers, batch 4, fp16 autocast). The ±512 flash-attention sliding window of
the baseline is timed as a reference.
"""

import torch
from anemoi.models.distributed.shapes import GraphShardInfo
from anemoi.models.layers.processor import TransformerProcessor
from anemoi.models.layers.utils import load_layer_kernels
from torch.nn.attention.flex_attention import flex_attention

from nese_training.radius_attention import RadiusTransformerProcessor
from nese_training.radius_attention import adjacency_mask

torch._dynamo.config.cache_size_limit = 64

TILE_64 = dict(BLOCK_M=64, BLOCK_N=64, BLOCK_M1=64, BLOCK_N1=64, BLOCK_M2=64, BLOCK_N2=64)
# (block_size, flex_attention kernel_options, torch.compile mode). In torch 2.8 a block size must be a
# multiple of the default kernel tile (128 x 64) unless max-autotune offers smaller candidate tiles.
AUTOTUNE = "max-autotune-no-cudagraphs"
CONFIGS = [
    (128, None, None),
    ((128, 64), None, None),
    ((128, 64), TILE_64, None),
    (128, None, AUTOTUNE),
    ((128, 64), None, AUTOTUNE),
    (64, None, AUTOTUNE),
    ((64, 128), None, AUTOTUNE),
]
BATCH_SIZE = 4
WARMUP, REPEATS = 10, 30
PROCESSOR = dict(
    num_layers=16,
    num_channels=512,
    num_chunks=1,
    num_heads=16,
    mlp_hidden_ratio=4,
    dropout_p=0.0,
    qk_norm=False,
    softcap=0.0,
    use_alibi_slopes=False,
    layer_kernels=load_layer_kernels(instance=False),
)

device = torch.device("cuda")
graph = torch.load("graph.pt", weights_only=False)
edge_index = graph["hidden", "to", "hidden"].edge_index.to(device)
num_nodes = graph["hidden"].num_nodes
shard_info = GraphShardInfo(nodes=[num_nodes])


def time_step(processor: torch.nn.Module, **kwargs) -> tuple[float, float]:
    """Mean ms per forward+backward and peak memory in GiB."""
    x = torch.randn(BATCH_SIZE * num_nodes, PROCESSOR["num_channels"], device=device, requires_grad=True)

    def step():
        with torch.autocast("cuda", dtype=torch.float16):
            out = processor(x, BATCH_SIZE, shard_info, **kwargs)
        out.float().square().mean().backward()

    for _ in range(WARMUP):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(REPEATS):
        step()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / REPEATS, torch.cuda.max_memory_allocated() / 2**30


def report(label: str, ms: float, gib: float, extra: str = "") -> None:
    print(f"{label:<50} {ms:8.2f} ms  {gib:6.1f} GiB  {extra}", flush=True)


print(torch.__version__, torch.cuda.get_device_name(), f"nodes={num_nodes}", flush=True)

torch.manual_seed(0)
window = TransformerProcessor(window_size=512, attention_implementation="flash_attention", **PROCESSOR).to(device)
report("sliding window ±512 (flash)", *time_step(window))
del window

for block_size, kernel_options, mode in CONFIGS:
    torch.manual_seed(0)
    processor = RadiusTransformerProcessor(
        attention_implementation="flex_attention", block_size=block_size, **PROCESSOR
    ).to(device)
    attention = processor.masked_attention
    attention.set_mask(adjacency_mask(edge_index, num_nodes))
    compiled = torch.compile(flex_attention, mode=mode)
    attention.flex_attention = lambda q, k, v, block_mask: compiled(q, k, v, block_mask=block_mask, kernel_options=kernel_options)
    label = f"radius flex {block_size} tile {kernel_options and kernel_options['BLOCK_M']} {mode or ''}"
    try:
        ms, gib = time_step(processor, edge_index=edge_index)
    except Exception as error:
        print(f"{label}: {type(error).__name__} {str(error).splitlines()[0]}", flush=True)
        continue
    finally:
        del processor
        torch.cuda.empty_cache()
    report(label, ms, gib, f"block sparsity {attention.block_mask.sparsity():.1f}%")
