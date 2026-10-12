# Training schedule sweep: all-altimeter, 24h hpx6

A short sweep over peak learning rate, weight decay and schedule length, to find the best
hpx6/24h model.
Its `last.ckpt` becomes the `warm_start` for the hpx8/6h fine-tune
(`../all-altimeter/training.stage2.yaml`).

The data is from all-altimeter, and this is based on v1 settings.

## Motivation

Validation loss bottoms out before the training schedule ends:
- total validation is best at epoch 10 (0.5481) and ends at 0.5493;
- sla is best at epoch 7 (0.2900) and ends at 0.2921;
- swh is best at epoch 9 (0.1824) and ends at 0.1837.

Training loss keeps falling over the same stretch, so the low-LR tail is fitting the training
years.
- The training loss rises a little just after warmup (steps 1440–2016, LR ≈ 9e-4), which
  suggests 1e-3 is near the top of the usable range.

## Sweep

Weight decay is not swept on its own. Following Wang & Aitchison (arXiv:2405.13698), AdamW's
weights are an exponential average over the last τ = 1/(η·λ) steps, where η is the LR AdamW
actually uses (4 × the config `lr`, since we use 4 GPUs and 1 GPU per model) and λ is the weight
decay (keeping with their notation).
They find the best τ in epochs
stays roughly constant, and for multi-epoch training lies between 1 epoch and the total number
of epochs. Here one epoch is 576 steps and a run is 12 epochs.

- v1 (η = 1e-3, λ = 0.1) has τ = 10,000 steps, about 17 epochs. That is longer than the whole
  run, so weight decay barely acts.
- A plain weight-decay grid based on that setup would hide this:
  at lr 6.25e-5, λ = 0.01 gives τ ≈ 700 epochs,
  which is effectively no decay.

### Round 1: Set 4 LRs and 3 τ values, compute λ:

Fix `max_steps` 6912 and `warmup_t` 500, then

| config lr (AdamW lr) | τ = 2500 (4.3 ep) | τ = 5000 (8.7 ep) | τ = 10000 (17.4 ep) |
|---|---|---|---|
| 6.25e-5 (2.5e-4) | λ = 1.6 | 0.8 | 0.4 |
| 1.25e-4 (5e-4) | 0.8 | 0.4 | 0.2 |
| 2.5e-4 (1e-3) | 0.4 | 0.2 | 0.1, which is v1 |
| 5e-4 (2e-3) | 0.2 | 0.1 | 0.05 |

all-altimeter-v1 corresponds to `runs/lr2.5e-4_tau10000_s6912` via symlink.

### Round 2: More steps (and warmup)

Take the winning LR and τ, try `max_steps` 3456 / `warmup_t` 250 and 13824 / 1000.
τ is held fixed, and so is λ. The paper keeps τ in epochs constant, and an epoch here is still
576 steps. τ just becomes a larger share of the short run and a smaller share of the long one.

Each run trains from scratch (`run_id` null) and logs to `runs/lr<lr>_tau<τ>_s<steps>/`.

### Scripts to run it

```bash
bash sweep.sh round1                # 11 jobs (v1 is borrowed), ~30 min each, 2 at a time on devel
bash sweep.sh round2 <lr> <tau>     # 2 jobs
module load nese && python summarize.py
```

`sweep.sh` skips any setting whose `runs/` directory already exists. A failed run leaves its
directory behind, so delete `runs/<label>` before resubmitting that setting. Logs are in
`slurm/<label>-<jobid>.out`.

Runs are ranked by **final** validation loss, because `last.ckpt` is what stage 2 starts from.
The best epoch is shown beside it in order to see overfitting.

Note: all-altimeter v0 and v1 differ by 0.0003, and although there are some
differences between the two, this essentially indicates the impact of
initialization noise

## Results

### Round 1

Ranked by final validation loss. Per-field columns are final values.

| run | λ | final val | best val (epoch) | sla | uvgxt | swh |
|---|---|---|---|---|---|---|
| **lr2.5e-4_tau2500_s6912** | 0.4 | **0.5463** | 0.5463 (12) | **0.2864** | 1.3010 | **0.1817** |
| lr5e-4_tau2500_s6912 | 0.2 | 0.5468 | 0.5468 (12) | 0.2873 | **1.2992** | 0.1839 |
| lr2.5e-4_tau5000_s6912 | 0.2 | 0.5470 | 0.5467 (11) | 0.2870 | 1.3005 | 0.1838 |
| lr1.25e-4_tau5000_s6912 | 0.4 | 0.5480 | 0.5480 (11) | 0.2891 | 1.3011 | 0.1841 |
| lr1.25e-4_tau10000_s6912 | 0.2 | 0.5481 | 0.5474 (10) | 0.2888 | 1.3010 | 0.1848 |
| lr5e-4_tau5000_s6912 | 0.1 | 0.5485 | 0.5481 (11) | 0.2888 | 1.3018 | 0.1851 |
| lr5e-4_tau10000_s6912 | 0.05 | 0.5485 | 0.5480 (11) | 0.2881 | 1.3008 | 0.1867 |
| lr6.25e-5_tau10000_s6912 | 0.4 | 0.5488 | 0.5488 (11) | 0.2896 | 1.3033 | 0.1839 |
| lr2.5e-4_tau10000_s6912 (v1) | 0.1 | 0.5493 | 0.5481 (10) | 0.2921 | 1.3023 | 0.1837 |
| lr6.25e-5_tau5000_s6912 | 0.8 | 0.5497 | 0.5497 (12) | 0.2897 | 1.3053 | 0.1848 |
| lr1.25e-4_tau2500_s6912 | 0.8 | 0.5510 | 0.5510 (12) | 0.2905 | 1.3063 | 0.1869 |
| lr6.25e-5_tau2500_s6912 | 1.6 | 0.5608 | 0.5608 (12) | 0.2979 | 1.3200 | 0.1965 |

- Winner: lr 2.5e-4, τ 2500 (λ 0.4), 0.0030 below v1 (10× the noise floor). The top three are
  within 0.0007, about 2× noise; the lower-lr rule picks the same run.
- At lr ≥ 2.5e-4, τ 2500 removes v1's late drift: best epoch is the final one. τ 10000 runs
  still peak at epoch 10–11.
- The best τ is not lr-independent: τ 2500 is worst at lr 6.25e-5 and 1.25e-4, where the runs
  look undertrained.
- τ 2500 is the grid edge for both high lrs, so a shorter τ (≈1250, ~2 epochs) is untested.
- sla and swh carry the gain over v1; uvgxt is nearly flat across the sweep.

### Round 2

lr 2.5e-4, τ 2500 (λ 0.4) at three schedule lengths.

| run | epochs | final val | best val (epoch) | sla | uvgxt | swh |
|---|---|---|---|---|---|---|
| lr2.5e-4_tau2500_s3456 | 6 | 0.5530 | 0.5530 (6) | 0.2919 | 1.3105 | 0.1878 |
| **lr2.5e-4_tau2500_s6912** (round 1) | 12 | **0.5463** | 0.5463 (12) | **0.2864** | 1.3010 | **0.1817** |
| lr2.5e-4_tau2500_s13824 | 24 | 0.5483 | 0.5480 (20) | 0.2899 | 1.3010 | 0.1842 |

- The 12-epoch run stays the overall winner. The 6-epoch run is undertrained (+0.0067).
- The 24-epoch run peaks at epoch 20 and drifts after it. Even its best epoch (0.5480) is behind
  the 12-epoch final, so the extra length goes to fitting the training years.
