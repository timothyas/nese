# Cross Track Surface Geostrophic Velocities

## TODO:

* Isolate loss per variable
* Visualize velocity fields
* Retest with u/v as inputs
* review latent sizes
* after this is working, make another version with no additional forcing inputs to see what the
  velocity diagnostics provide on their own
* evaluate the error using the gridded `mdt_u/v` versus the along track product

---

Picking up from `field-encoder`, this experiment tests the impact of including
surface geostrophic velocities.
Because these fields are tied to SLA, we design the model to diagnose the
quantity via a physics-diagnostics block that comes at the very end of the model
architecture.

There are two key components that this adds to the code base

### 1. The model block

The necessary code to `nese_training` to compute `ugxt` and `vgxt`.
The `xt` denotes where the fields come from, indicating that they can more readily compare to the
observed cross-track (xt) velocities, but these will be full fields rather than
along track.
The only points with NaNs that are unique compared to SLA are:
* MDT is NaN even though SLA is not
* +/- 5 degrees from the equator, due to the coriolis term
* points with fewer than `min_nearest_neighbours`, at which the gradient
  computation is poorly defined. Default is 3 but it's a hyperparameter.
This computation follows the procedure used in `nese-dataprep`, see
the `sources.cmems_l3_sla` module (also some notes in
`nese-dataprep/docs/data_sources.md` on this).

I want the code block to look like a natural part of the anemoi framework, added
to the model block of the yaml file, something like:

```yaml
model:
  physics_diagnostics: ["uvgxt"]
```

That shorthand should be enough, and we should check early that we have all the
inputs.
The required inputs are:
* model computed SLA
* For now `mdt_u` and `mdt_v`, but in the future we could also take `mdt` and
  compute the derivatives ourselves

Note that we may or may not provide the mdt fields as input to the model
separately, but that doesn't matter, we want them in their data-space (not a
latent representation) here.


### 2. The loss function

Although we may not provide ugxt and vgxt as inputs, we do want to score the diagnosed
ugxt and vgxt against the observations.
So, they do need to be in the dataset just as targets.
Additionally, we may need to rotate the full-field velocities so that they are
consistent with the cross track observations.
So, this will probably need special treatment in the loss function, maybe by
searching for "xt" in the variable name, or maybe it needs to be more careful
than that.

### Implementation

See "Physics diagnostics" in `nese/training/README.md`.
The loss's `observations` gains `uvgxt: {components: [ugxt, vgxt]}`. The same spec pools
the normaliser and imputer statistics and composes the plotted truth.

## Version 0

* `mdt`, `mdt_u`, and `mdt_v` are provided as static forcing inputs
* `ugxt` and `vgxt` are not provided as inputs
    * This is why the field encoder section in training.yaml
      gets its own `fields` instead of the loss reference

Wall-clock time = 28 minutes for 12 epochs (6912 iterations). Same as
`field-encoder`.
Error is significantly worse than field-encoder in the relative sense,
(although still just 1mm difference).
Is this overfitting sooner, and more dramatically?

```
Diagnose-uvgxt relative to field-encoder-v2, pooled over groups
---------------------------------------------------------------
(time mean of |metric|; negative is an improvement)

                            RMSE                       MAE                      Bias
Unseen    0.06183 ->  0.06264 (  +1.3%)    0.03717 ->  0.03773 (  +1.5%)    0.00173 ->  0.00178 (  +2.8%)
+0h       0.04220 ->  0.04306 (  +2.0%)    0.02629 ->  0.02702 (  +2.8%)    0.00051 ->  0.00054 (  +5.5%)
-24h      0.04329 ->  0.04421 (  +2.1%)    0.02695 ->  0.02770 (  +2.8%)    0.00056 ->  0.00060 (  +6.0%)
-48h      0.04339 ->  0.04449 (  +2.5%)    0.02711 ->  0.02799 (  +3.2%)    0.00062 ->  0.00060 (  -2.3%)
```
