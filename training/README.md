# nese-training

Observation-supervised sea level anomaly state estimation, built on anemoi-core.

Per-mission altimetry observations, binned onto a HEALPix L6 mesh by `nese-dataprep`,
go in; one dense `sla` field comes out.

## The task

Observations are bin averages at discrete timestamps. Estimating the field at `t` from
a window that *includes* bin `t` gives the loss no gradient at unobserved cells --
which is the entire quantity of interest -- so the model would be unconstrained exactly
where it matters. Supervising against the *next* bin avoids that: satellite ground
tracks shift between bins, so roughly **55% of the target bin's observed cells are
unobserved in the two input bins** (measured on this dataset). That held-out
supervision is free.

So the task is `t-24h, t -> t+24h`, which is stock `anemoi.training.tasks.Forecaster`
with `rollout.maximum: 1`. At that setting `steps()` yields a single element, the
`if i < len(task_steps) - 1` guard in `train/methods/single.py` never fires, and
`advance_input` -- the only forecast-specific code in the task -- never executes.

The cost: this is a one-bin forecast, not an analysis. The model cannot use
observations at the estimate time, and the residual confounds interpolation error with
24h of genuine evolution.

## Structure

Every channel carries an explicit role, so `prognostic` is empty and model input /
model output are fully disjoint:

| role | channels |
|---|---|
| `forcing` (model input) | `sla_filtered_*`, `obs_count_*` |
| `diagnostic` (model output) | `sla` |
| `target` (loss only) | `sla_filtered_*`, `obs_count_*` |

`sla` is a duplicate of one mission's column. Under the combined loss its values are
never a target -- only its statistics reach the normaliser -- so how missions get
merged is still an open question, deferred.

Only three missions flew in 2000-2001 (`ers_2`, `gfo`, `topex_poseidon`); the other 26
have NaN statistics over this window and are excluded.

## Running

```bash
export NESE_DATA_DIR=/Users/timith/work/nese/observations/sea-level/altimetry/data/anemoi-hpx6
export NESE_RUN_DIR=$PWD/runs/my-run
export KMP_DUPLICATE_LIB_OK=TRUE   # see Local quirks

python scripts/build_graph.py      # optional; training builds it on demand
anemoi-training train --config-dir=$PWD/config --config-name=nese
```

Tests: `pytest tests/` -- these check the masking claim in `CountMaskedMSE` directly,
which otherwise fails silently.

## Config

`config/nese.yaml` is a single flattened file, produced by `anemoi-training config dump`
from what were previously twelve group files, and it is now the source of truth. There
is no `defaults:` list; the anemoi packaged defaults are inlined.

The consequence worth remembering: **it no longer inherits upstream default changes.**
On a develop branch that is a real trade -- exact reproducibility in exchange for
silent drift from anemoi-core. After an anemoi-core update, diff this file against a
fresh dump of the stock recipe rather than assuming it is current.

CLI overrides still work as normal:

```bash
anemoi-training train --config-dir=$PWD/config --config-name=nese \
    dataloader.limit_batches.training=4 training.max_steps=100
```

Note that `anemoi-training config validate` **ignores** `config_validation: false` by
design (`commands/config.py:96`), so it will still report `CountMaskedMSE` as invalid
against the closed loss enum. That is expected, not a broken config. `config dump` does
not validate at all.

## Field encoder

`nese_training.field_encoder.FieldEncoderEncProcDec` is optional (select it with
`model.model._target_`). Instead of one input channel per mission, it maps each
field's observations through one shared MLP (`1 -> field_channels`) and averages
over the groups that observed each cell (`count > 0`, the same pooling as the loss).
Unobserved cells get a zero latent. The field latents replace the `{observed_as}_*`
and `count` columns in the input concat; all other inputs pass through unchanged.

```yaml
model:
  model:
    _target_: nese_training.field_encoder.FieldEncoderEncProcDec
    fields: ${training.training_loss.datasets.ocean.observations}
    field_channels: 32
```

A vector field (`components: [ugxt, vgxt]`) passes each group's components through its
MLP together (`n_components -> field_channels`), so each observed vector maps to a single latent.
A diagnosed field's observations can be inputs like any other field, while the diagnosed field itself
stays a `diagnostic` output. `defined_by: <physics diagnostic>` also masks the field to the cells
where that diagnostic is defined, because elsewhere its observations are NaN and have been imputed:

```yaml
uvgxt:
  components: [ugxt, vgxt]
  defined_by: uvgxt   # ignored by the loss and processors
```

A bin can still be imputed with `count > 0` outside that set, when none of a mission's observations
in the bin have a velocity (e.g. single-observation passes in `nese-dataprep`).

## Physics diagnostics

`model.physics_diagnostics` lists model outputs computed from other model outputs rather
than by the network (`nese_training.physics`). Currently available options are:

```yaml
model:
  physics_diagnostics: [uvgxt]   # or [{uvgxt: {min_neighbours: 3}}]
```

### Surface Geostrophic Velocity

`uvgxt` diagnoses the full-field surface geostrophic velocity from the predicted `sla`,
`ugxt = mdt_u - (g/f) d(sla)/dy` and `vgxt = mdt_v + (g/f) d(sla)/dx`. Constants, names and
the 5 degree equatorial band follow `nese-dataprep`'s `cmems_l3_sla`. The derivatives are
a least-squares fit to each HEALPix cell's retained neighbours. The velocity is undefined
within the equatorial band and at cells keeping fewer than `min_neighbours` (default 3), as
on some coasts. `physics.geostrophy_defined` is the place where this set is
defined. There, the
block outputs only the mean flow (or zero, with no MDT), the loss doesn't score the cell, and
`processors.GeostrophyMaskPostprocessor` sets it to NaN on the way out, in both the
prediction and the composed truth. Give the loss and the postprocessor the same
`min_neighbours` by interpolation, as `experimental/diagnose-uvgxt/training.yaml` does.

* `ugxt` and `vgxt` must be `diagnostic` model outputs, never inputs. This is checked at
  build time, along with `sla` being an output and `mdt_u`/`mdt_v` being in
  `system.input.dataset`.
* The decoder emits only the network's outputs, so the network never predicts a diagnosed
  field itself.
* `mdt_u`/`mdt_v` are read in raw units from the store and saved in the checkpoint.
  Listing them as forcing is optional.
* The block uses the scales of the dataset's own normalizer, built from the same config.
* The grid can't be sharded, since the gradient needs every cell's neighbours.

The observations are cross-track only, so the loss scores the vector field by its
component across the observed track:

```yaml
observations:
  uvgxt:
    components: [ugxt, vgxt]   # observed as {component}_{group}
```

The track normal is the observed vector's direction in raw units, so the loss needs
`dataset: ${system.input.dataset}` to pool the statistics the same way the normaliser does.
A vector field's components share one pooled stdev, which keeps directions intact in
normalised space, and each keeps its own pooled mean. The loss skips cells where the
velocity is undefined and cells with zero observed speed. A bin crossed by tracks of
different orientation only approximates a projection; see `nese-dataprep`'s `docs/data_sources.md`.

## Why the loss is custom

`CountMaskedMSE` needs a per-mission coverage mask at the target time. The existing
machinery cannot supply one:

- `NaNMaskScaler` reads `loss_mask_training` off the imputer, but
  `preprocessing/imputer.py:210` builds it from `nan_locations[:, 0]` -- timestep 0,
  the oldest input bin, not the target bin. Satellite coverage changes every bin.
- Scaler callbacks receive `(model, dataset_name)` and never the batch.
- Scalers are keyed on model-output variables, of which there is exactly one.

So the mask rides in on the target tensor instead: each loss term is configured with
paired variables, `target_variables: [sla_filtered_X, obs_count_X]`. Counts use `max`
normalisation (`x / maximum`, no additive shift), so a zero count stays zero and
`count > 0` remains exactly "this mission observed this cell in this bin".

## CPU vs the Mac GPU (MPS)

Training runs on **CPU** (`system.hardware.accelerator: cpu`). MPS works but is not
worth switching to, measured on this machine:

| | wall (8 steps) | per step |
|---|---|---|
| CPU | 36s | 1.12 it/s |
| MPS | 40s | 1.16 it/s |

A ~3% per-step gain, swamped by longer startup. Apple's unified memory means both
share bandwidth, and at this model size the cost is in graph gather/scatter and
dataloading rather than the dense math MPS accelerates. MPS also caps out around
20 GiB, and the L5 latent mesh (12288 nodes) OOMs there -- the numbers above are the
L4 mesh (3072), which fits. So the larger configurations where a GPU might actually
pay are exactly the ones that do not fit.

Two things block MPS out of the box, both handled by `nese_training.strategy`:
`train.py:566` does not accept `mps` as an accelerator (use `auto`, which Lightning
resolves to MPS), and Lightning refuses any DDP-family strategy on MPS. It is opt-in
and does not affect the default CPU path:

```bash
anemoi-training train --config-dir=$PWD/config --config-name=nese \
    system.hardware.accelerator=auto \
    training.strategy._target_=nese_training.strategy.SingleDeviceGroupStrategy \
    +training.strategy.device=mps \
    graph.nodes.hidden.node_builder.resolution=4
```

Delete `src/nese_training/strategy.py` if you never want this; nothing else uses it.

Note `diagnostics.checkpoint.every_n_epochs.num_models_saved: -1` keeps **every** epoch's
checkpoint, at ~59 MB each. A 25-epoch run wrote 17 GB. Set a finite number for real runs.

## Local quirks

- `config_validation: false` -- `training.training_loss._target_` is validated against
  a closed enum of built-in losses, so `CountMaskedMSE` cannot pass. The task does not
  need this escape hatch; stock `Forecaster` validates fine. Validation is
  all-or-nothing. A small upstream PR adding the loss to that enum would restore it.
- `KMP_DUPLICATE_LIB_OK=TRUE` -- pip-installed torch and conda-installed scipy each
  link their own OpenMP runtime. Local Mac only.
- `scripts/cast_counts_to_f32.py` -- `nese-dataprep` writes counts as int32, and
  numpy promotes int32 + float32 to float64. anemoi overrides
  `transfer_batch_to_device` to move but not cast, so the model receives a float64
  batch and the first `Linear` fails. **The permanent fix belongs in nese-dataprep.**
- The latent mesh is HEALPix with no `hidden->hidden` edges, because the processor is a
  sliding-window transformer. `HEALPixMultiScaleEdges` is also broken upstream:
  `graphs/generate/healpix.py:14-21` keeps the `-1` healpy returns for the 8 corner
  pixels with only 7 neighbours, and the multi-scale rescaling turns those into large
  negative indices.
- `validation_metrics` is empty. Per-variable metrics subset scalers along the variable
  dimension against `model.output` (one variable), and the paired target layout gives
  the mapper two columns, so the subsetting runs off the end. The validation loss is
  the same `CountMaskedMSE` and is still reported.

## Status

Runs end to end on CPU. On 4 fixed samples the loss falls 3.52 -> 1.71 over 100 steps,
so gradient flows. That is a plumbing check, not skill: standardised SLA has unit
variance, so predicting the mean scores about 1.0, and the model has not yet beaten
that trivial baseline. Beating 1.0 on held-out bins is the first real target.
