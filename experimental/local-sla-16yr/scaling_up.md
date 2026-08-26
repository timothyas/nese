# Scaling up: CPU laptop → GPU/HPC

Things that are true on the laptop but will change, break, or start to matter on the HPC
machine. Notes accumulate here as they are found; nothing in this file is a to-do unless it
says so.

## Landmines

### `Boolean1DMask` builds its mask on the wrong device

`model.output_mask` is a `Boolean1DMask` keyed on the `sea_mask` node attribute. Its
`apply()` forces the broadcast mask to `.cpu()` before `masked_fill`:

```python
mask = self.broadcast_like(x, dim, grid_shard_slice).cpu()
return Boolean1DMask._fill_tensor_with_float(x, ~mask, fill_value)
```
`anemoi/training/utils/masks.py:121-122`

but `graph_data` is moved to the accelerator first (`train/methods/base.py:192`), so the
loss-scaler path (`losses/scalers/node_attributes.py:64`) would call `masked_fill` with a
CPU mask against a CUDA tensor and raise a device mismatch.

This cannot fire on `accelerator: cpu`, so it will appear for the first time on GPU, at
startup, in the scaler construction. Likely a one-line upstream fix. **Check this early.**

## Sizing reference

Graph at hpx6 → hpx5, 200 km cutoff, KNN 3:

| | count |
|---|---|
| ocean nodes | 49,152 (33,469 sea, 15,683 land) |
| hidden nodes | 12,288 |
| encoder edges (ocean→hidden) | 121,053 |
| decoder edges (hidden→ocean) | 147,456 |

**Edges, not grid points, set the memory ceiling.** One edge-message tensor in the encoder
is roughly `121k × 128 channels × 4 B` ≈ 62 MB *per batch element*, and autograd keeps
several of them plus gradients.

Trainable embeddings at `trainable_parameters: 2` — 98,304 data + 24,576 hidden + 242,106
data2hidden + 294,912 hidden2data ≈ **660k parameters**, each carrying two AdamW moment
buffers. Note 15,683 × 2 = 31,366 of those sit on land nodes that receive no gradient once
the output mask is on; they are still allocated, still in the optimizer state, and still
decay under AdamW's default `weight_decay=1e-2`.

Training set is 5,479 daily samples. At `batch_size.training: 8` that is 685 batches per
epoch. `max_steps` counts **optimizer** steps, so if batch size has to come down for memory,
`accum_grad_batches` restores the effective batch and preserves the epoch count for free.

## Still open

- **Encoder/decoder gradient checkpointing.** `processor.gradient_checkpointing` is on;
  `encoder` and `decoder` are still `false`. Since the edge tensors are the ceiling, those
  two are where checkpointing buys the most.
- **Plot frequency.** `diagnostics.plot.frequency.batch: 5` drives three callbacks — about
  137 plot events per epoch at bs=8, each rendering a 49,152-cell global field. Note
  `every_n_batches` counts batches, not optimizer steps, so gradient accumulation multiplies
  it.
- **`Boolean1DMask.as_tuple()` asserts a contiguous block of True.** The sea mask is not
  contiguous. Currently unreachable — it is only called when a loss config contains the
  literal value `subgrid: output_mask` (`train/methods/base.py:1236-1237`) — but it rules
  out the spectral losses until that assert is relaxed upstream.
- **`output_mask` in the checkpoint activates LAM boundary forcings.** Inference reads the
  `output_mask` supporting array only to gate `BoundaryForcings`
  (`anemoi/inference/tensors.py:361`), which *overwrites* masked points from a driving
  dataset. Inert today because `data.output.prognostic` is empty, so no boundary forcings
  are built — but it becomes live the moment a prognostic variable is added.
