# Throughput sweep: hpx8/6h, 4 GPUs

Finds the per-GPU batch size, `num_workers` and `prefetch_factor` for the stage 2 fine-tune
(`../all-altimeter/training.stage2.yaml`): what fits in GPU memory, how fast it runs, and how
many workers keep the GPUs fed.

`training.yaml` is `../all-altimeter/training.stage2.yaml` with these changes:
- the dataloader dates fit the 2000–2005 store: training 2000–2004, validation and test 2005;
- 200 training and 10 validation batches per run;
- no plots, no checkpoints, and the profiler's memory trace, snapshot and model summary are off;
- 2 validation workers instead of 8, to save host memory;
- Lightning's `DeviceStatsMonitor` logs peak GPU memory. mlflow system metrics are also on, but
  they only record one sample at startup. `summarize.py` reads peak host memory from the
  Slurm log instead;
- `update_ds_stats_on_ckpt_load.states: false`, see below.

Each job runs `anemoi-training profile`, which reports throughput from iteration 100 onwards
(after the flex_attention compile) and the mean wait on the training dataloader.

## The 2000–2005 store's NaN statistics

Groups that were not in orbit during 2000–2005 are all NaN in this store. That includes jason_2,
which the dataloader join renames to sla/ugxt/vgxt/swh. These are 106 of the 158 variables, and
their statistics are NaN or have zero std, so a normaliser recomputed here would turn every input
into NaN. The variables and their order match the hpx6 store, so these runs keep v2's normaliser
from the warm start instead. The groups stay in, so the cost per sample matches the full
1993–2020 store, where every group has data.

## Size of one sample

One timestep is one zarr chunk: 158 variables × 548,379 cells, 347 MB as float32 (~8.6 MB
compressed). A sample is 4 timesteps. The jason_2 join reads the same store a second time, so a
sample is about 1.4–2.8 GB decompressed. Host memory grows with
4 ranks × workers × prefetch × batch size × that.

## Sweep

```bash
bash sweep.sh round1              # bs 1, 2, 4 at 8 workers, prefetch 2; bs1 builds graph.pt first
bash sweep.sh round2 <bs>         # workers 2, 4, 8, 16 × prefetch 1, 2 at the chosen bs
bash sweep.sh round3 <bs> <w> <pf> # flex_attention block size 32, 64, 128, 256
bash sweep.sh one <bs> <w> <pf> [blk]  # a single setting
module load nese && python summarize.py
```

The bs1 job also serves as the smoke test. bs2 and bs4 wait for it (`afterok`). If it fails,
they stay pending with `DependencyNeverSatisfied`; `scancel` them, fix the problem, and delete
`runs/<label>` before resubmitting. An OOM is a valid result, reported as "OOM" by
`summarize.py`.

How to choose:
- **batch size:** the highest samples/s per GPU that leaves some GPU memory headroom. A larger
  batch also means fewer steps per epoch, which changes τ = 1/(η·λ) in epochs.
- **workers:** the fewest that keep the dataloader wait near 0 s.
- **prefetch 2:** only if it is clearly faster than prefetch 1 at that worker count, given the
  extra host memory.
- **block size:** the fastest one that compiles. Use it for `model.processor.block_size` in
  `../all-altimeter/training.stage2.yaml`.

## Round 3: flex_attention block size

`model.processor.block_size` is the tile size of the processor's flex_attention BlockMask. The
processor attends within 1200 km on 49,152 hpx6 hidden nodes. With nested HEALPix ordering,
smaller tiles skip more empty space but keep the tensor cores less busy. Only speed and memory
change, not the output. Round 3 runs after round 2, so the dataloader is no longer the
bottleneck and step time reflects the attention kernel. `training.yaml` uses 64, so the blk64 run
repeats a round 2 setting and gives an estimate of run-to-run noise. Block size 32 may fail to
compile with `max-autotune-no-cudagraphs`; that counts as a result.

## Results

GPU memory is rank 0's peak (allocated / reserved) of 95 GB. Host memory is the job's peak. The
s.d. column is the standard deviation of the training rate over iterations 100–200, and is a
better sign of a loader bottleneck than anemoi's mean wait, which stays small even when the
loader is the bottleneck.

### Round 1: batch size (8 workers, prefetch 2)

| run | result | peak GPU (GB) | peak host (GB) |
|---|---|---|---|
| bs1_w8_pf2 | 1.91 samples/s/GPU | 54.4 / 59.2 | 323 |
| bs2_w8_pf2 | host OOM in sanity check | – | 514 |
| bs4_w8_pf2 | host OOM in sanity check | – | 591 |
| bs2_w4_pf1 (retry, lighter loader) | GPU OOM: 93.9 of 95 GB | – | 175 |

The first bs1_w8_pf2 run used 8 validation workers and peaked at 413 GB host. It was rerun in
round 2 with the current config, and the row above shows the rerun.

- Batch size 1 is the only size that fits without gradient checkpointing. bs2 would need roughly
  110 GB of GPU memory.
- The host runs out of memory at ~500 GB, well below the 808 GB requested.

### Round 2: workers × prefetch (bs 1)

| workers | prefetch | samples/s/GPU | step (s) | rate s.d. | peak host (GB) |
|---|---|---|---|---|---|
| 2 | 1 | 0.88 | 1.14 | 0.36 | 132 |
| 2 | 2 | 1.13 | 0.88 | 0.69 | 168 |
| 4 | 1 | 1.46 | 0.69 | 0.33 | 162 |
| 4 | 2 | 1.67 | 0.60 | 0.46 | 217 |
| **8** | **1** | **1.91** | **0.52** | **0.07** | **236** |
| 8 | 2 | 1.91 | 0.52 | 0.07 | 323 |
| 16 | 1 | 1.92 | 0.52 | 0.07 | 381 |
| 16 | 2 | 1.85 | 0.54 | 0.23 | 512 |

GPU memory is 54.4–54.6 / 59.2 GB in every run.

- Choice: **8 workers, prefetch 1**. From 8 workers up the GPU is the bottleneck (1.91 samples/s
  per GPU), and prefetch 2 only costs host memory.
- One worker loads a sample in ~2.3 s (from the 2-worker runs), so 8 workers supply ~3.5
  samples/s per GPU, 1.8× what the GPU uses. Revisit if the step drops below ~0.3 s.
- Prefetch only helps when workers are short (2, 4). At 16 workers prefetch 2 is slightly slower,
  probably from host memory pressure.
- Stage 2 at these settings: ~7.6 samples/s per node, so one epoch of 1993–2018 (~36,800
  samples, ~9,200 steps) takes ~80 min, and 12 epochs ~16 h.

### Round 3: block size (bs 1, 8 workers, prefetch 1)

Submitted 2026-10-13, jobs 2511990–2511993.

| blk | samples/s/GPU | step (s) | rate s.d. | peak GPU (GB) |
|---|---|---|---|---|
| 32 | compile fails: Inductor `NoValidChoicesError`, no kernel for 32-wide tiles | – | – | – |
| **64** | **1.92** | **0.52** | 0.07 | 54.6 / 59.2 |
| 128 | 1.84 | 0.54 | 0.07 | 54.4 / 59.2 |
| 256 | 1.81 | 0.55 | 0.07 | 54.6 / 59.2 |

- Keep **64**, the current setting. 128 is 4% slower and 256 is 6% slower. GPU memory is the
  same for all three.
- Run-to-run noise is small: the blk64 run repeats round 2's 8-worker, prefetch-1 setting, and both
  give 1.915 samples/s. So the differences above are real, even though they are small. Block size
  only changes the processor's attention kernel, which is a fraction of the step.

### Verification: hpx8/24h

The run moved to 24h (`../all-altimeter/anemoi-hpx8-24h`, 1993-11-01 → 2020-12-31, 94 GB). It has
the same cells and variables as the 6h store, so `graph.pt` is reused. Every mission has data in
its statistics period, so the normaliser is recomputed from this store, as in the real run.
`bash sweep.sh verify24h` runs the chosen settings (bs 1, prefetch 1, block size 64) at 4 and 8
workers, with the stage 1 date split. Bins are 4× longer, so each sample has more observed cells
but the same dense size. Speed should not change; the 4-worker run would show it if loading got
slower.

| workers | samples/s/GPU | step (s) | rate s.d. | peak GPU (GB) | peak host (GB) | val loss (sla / uvgxt / swh) |
|---|---|---|---|---|---|---|
| 4 | 1.49 | 0.67 | 0.28 | 54.4 / 59.2 | 166 | 0.532 (0.211 / 1.302 / 0.214) |
| **8** | **1.90** | **0.53** | 0.07 | 54.4 / 59.2 | 239 | 0.532 (0.209 / 1.306 / 0.212) |

The settings hold at 24h: 8 workers keep up and speed matches the 6h runs (1.91).

The first two attempts (jobs 2512417–2512418, then 2512771–2512772) gave NaN for ~2/3 of training
steps and every validation step. 16-mixed's grad scaler skips the steps with non-finite
gradients, so training only looked slow to learn. The cause was in `ObservationMaskedMSE`: at
hpx8/24h a field's observed (mission, cell) pairs per bin pass float16's 65504 (sla: ~70k in
2017, ~85k in 2019; at most ~18k at hpx6/24h or hpx8/6h). The per-field coverage matmul runs in
float16 under autocast, overflowed to inf, and inf × 0 in `field_membership` turned every column
into NaN. Only post-2016 samples, when most missions fly, were affected.

- Converting to float32 inside the loss was not enough: autocast still runs the matmul in float16.
- The fix runs `calculate_difference` with autocast off and in float32 (`_in_float32` in
  `training/src/nese_training/losses.py`). It has a CUDA test under autocast.
- Two diagnostics found it and stay in place. `nese_training.diagnostics.NonFiniteLossReporter`
  (enabled in this `training.yaml`) showed that the batch and the prediction were finite. The
  loss's own report named the intermediate that went bad (field coverage = inf).
- Store vs checkpoint statistics made no difference (jobs 2512563–4). A 32-bit run crashed with an
  illegal memory access in the compiled flex_attention kernel and was not pursued.
- Final runs: jobs 2513515–2513516, after the tests in 2513514 passed. Logs from the failed attempts
  are in `slurm/old/`.
