# Experiments

Lineage
1. `local-sla-02yr`: get the initial graphs and data pipelines working
2. `local-sla-16yr`: small scale up to uncover some issues. Also some small
   tests noted in the README, like adding more static fields, removing
   graph-specific features. None of that really mattered.
3. `olivia-sla-baseline`: baseline model on olivia
    * v0 through v1: some indication of spread per training run
    * v2: updated loss from per-live-mission loss to obs-count weighted
4. `dropout-sla`: use `dropout_p=0.2` during training.
    * Gains probably insignificant compared to per-training spread.
    * builds on `olivia-sla-baseline-v0`
5. `radius-processor`: use a radius-based processor instead of the sliding
   window
    * v0: live-mission loss
    * v1: obs-count loss and reduced `max_steps = 6912` (see below)
    * compare to `olivia-sla-baseline-v2`
6. `field-encoder`: encode fields to a track-agnostic latent state with a shared
   MLP
    * v0: obs-count loss for 10k steps
    * v1: reduced to 6912 steps
    * compare to `radius-processor`
    * v2: for future comparison, also bring in `ers_1` and `ers_1_geodetic`
    * v3: try `field_channels = 8` not 32, no big difference. Useful for bigger
      models
7. `diagnose-uvgxt`: add a `physics-diagnostics` module and compute surface
   geostrophic velocities from SLA
    * note the model computes velocity pointed in east/north directions, `xt`
      is still used in nomenclature to indicate where the fields come from
    * v0: no uvgxt as inputs, just computed and data used as targets
    * v1: add per-variable loss logging
    * v2: add uvgxt data inputs, one field encoder per vector pair, so now two
      MLPs, one for `sla` and one for `uvgxt`
    * v3: consistent velocity masking by providing a new obs count on the
      dataprep side, done per binned cell center, not obs location


## Overfitting

These runs are still simple univariate prototypes, so this isn't worth a ton of
attention, but even 10k iterations results in overfitting, at least with a peak
learning rate of 1e-3.
The table documents the number of steps it takes for the validation loss to
bottom out, and what the best value is.
It's all approximate though, since validation loss is recorded per-epoch only,
and the loss formulation changes as noted above.


| Run | Flattens around | Val loss there | Best / final |
|---|---|---|---|
| olivia-sla-baseline v1 | ~6,900–7,500 | 0.2743 → 0.2734 | 0.2727 @ 8,063 / 0.2740 |
| olivia-sla-baseline v2 | ~6,300–6,900 | 0.2739 → 0.2729 | 0.2724 @ 8,063 / 0.2742 |
| radius-processor v0 | ~6,900 | 0.2756 | 0.2756 @ 6,911 / 0.2773 |
| radius-processor v1 * | ~5,200–5,800 | 0.2767 → 0.2756 | 0.2756 @ 5,759 / 0.2762 |
| dropout-sla v0 | ~7,500 | 0.2718 | 0.2715 @ 9,215 / 0.2719 |
| field-encoder v0 | ~5,800–6,900 | 0.2762 → 0.2748 | 0.2748 @ 6,911 / 0.2799 |
| field-encoder v1 * | ~5,800–6,300 | 0.2760 → 0.2759 | 0.2759 @ 6,335 / 0.2764 |
| field-encoder v2 * | ~4,600–5,800 | 0.2738 → 0.2735 | 0.2735 @ 5,759 / 0.2747 |
| field-encoder v3 * | ~5,200–5,800 | 0.2772 → 0.2761 | 0.2761 @ 5,759 / 0.2772 |
| diagnose-uvgxt v0 * † | ~4,000–5,200 | 0.7495 → 0.7476 | 0.7476 @ 5,183 / 0.7511 |
| diagnose-uvgxt v1 * † | ~4,600–5,200 | 0.7488 → 0.7483 | 0.7483 @ 5,183 / 0.7516 |
| diagnose-uvgxt v2 * † | ~4,000–5,200 | 0.7445 → 0.7440 | 0.7440 @ 5,183 / 0.7469 |

`*` = Shortened to `max_steps: 6912` (12 epochs) rather than 10k.

`†` = Aggregate of the `sla` and `uvgxt` field losses (fields weighted equally), so not
comparable to the SLA-only rows above.
