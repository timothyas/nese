# Masking, imputing and processing in nese_training

Where data is masked, filled or transformed on its way through a training step, in order.

The network, the loss and backpropagation all work in normalised space. Pre-processors run on the
batch before it enters the network. Their inverse, the post-processors, run on model outputs
after the network, only for plots and inference. Neither is part of the decoder or of the
backward pass, so no gradient flows through a processor. The one exception is the physics block
(stage 3), which converts between normalised and physical units inside the model's forward pass.

## 1. Pre-processors: forward pass, on the batch, before the model

During training these run in `on_after_batch_transfer` (anemoi-core
`training/.../train/methods/base.py`). At inference they run inside `predict_step`
(`models/.../models/base.py`). The data carries no gradient at this point. In config order:

| processor | forward pass | inverse pass |
|---|---|---|
| `geostrophy_mask` (`GeostrophyMaskPostprocessor`) | does nothing | sets ugxt/vgxt to NaN where the geostrophic velocity is undefined (equatorial band, no MDT, too few neighbours) |
| `obs_nan_restorer` (`ObsCountNaNRestorer`) | does nothing | sets each value to NaN where its count is 0, and rebuilds the "truth" sla/ugxt/vgxt/swh as count-weighted means over missions |
| `imputer` (`PooledObservationImputer`) | fills NaN observations with the pooled field mean | does not undo its own fill on data-shaped tensors, and leaves the gaps to `obs_nan_restorer` |
| `normalizer` (`PooledObservationNormalizer`) | gives every mission of a field one shared mean and std; counts are divided by their maximum; static fields are min-max scaled | converts back to physical units |

The two inverse-only processors are listed first so that, with the order reversed on the inverse
pass, they run last, on values already back in physical units.

The pooled statistics come from `nese_training.observations.pool_statistics`. Each mission is
weighted by the dataset mean of its count channel. A mission with no observations in the
statistics period has a zero count and NaN statistics. It is left out of the pooled values, but
its columns still get the shared values.

The count channels are listed under both `forcing` and `target`. This puts them in the model
input, for the `FieldEncoder` (stage 2), and in `data.output`, for the loss (stage 4) and for
`obs_nan_restorer`.

## 2. Inside the network: input

`FieldEncoder` (`field_encoder.py`) passes each field through one MLP shared by all its missions.
In each cell it averages only over the missions whose count is above 0, and gives a zero latent
where no mission observed. So imputed fill values never reach the network as information. The
mask is taken separately for each input step from that step's counts. The counts are used only
as this mask and are left out of the other inputs, so the network never sees how many
observations a cell had.

## 3. Inside the network: output

`GeostrophicVelocity` (`physics.py`) diagnoses ugxt and vgxt from the predicted sla. It holds
the normaliser's scales (`normalisation_scales`): it converts sla to physical units, takes the
gradient, and converts the result back to normalised space. This is part of the forward graph,
so it is differentiated.

`output_mask` is `NoOutputMask`, which does nothing.

## 4. Loss: normalised space

`ObservationMaskedMSE` (`losses.py`):
- **Count mask:** each mission's term is scored only where that mission's count is above 0 in
  the *target* time bin.
- **Geostrophy mask:** the velocity term is scored only where the geostrophic velocity is
  defined. This uses the same `geostrophy_defined` function as the processor.
- **Velocity direction:** velocity is scored along each observed track. The track direction has
  to be in physical units, so the loss recomputes the pooled statistics from the store with
  `pool_statistics`, exactly as the normaliser does.
- **Pooling and weighting:** each field's loss is pooled over its missions as
  `sum(err**2) / sum(count > 0)`. Every observed (mission, cell) pair counts once, whatever its
  count. The fields are then combined with their weights and averaged over the fields observed
  in the sample.
- **float32:** at hpx8 the number of observed pairs in that sum can exceed float16's 65504, so
  the loss runs in float32 with autocast off (`_in_float32`).

anemoi's own NaN loss mask is deliberately not used. It builds its mask from the oldest *input*
bin, not the target bin.

## 5. Post-processors: inverse pass, on model outputs only

- **Plots:** the diagnostics callbacks (`training/.../diagnostics/callbacks/plot.py`) run the
  inverse chain on copies of the outputs. Plots are therefore in physical units, with gaps shown
  as NaN.
- **Inference:** `predict_step` does the same, so nese-inference writes fields in physical units.
- **Validation metrics:** none are configured, so validation loss is also computed in normalised
  space.

## Static masks, built once

These are not processors:
- **Ocean cells:** nese-dataprep's `select_cells` keeps only cells with land fraction below 0.1
  and a valid MDT.
- **Geostrophy-defined cells:** `store_geostrophy_defined` computes this mask once from the
  store.
