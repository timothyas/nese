# All Altimeter

Extends `diagnose-uvgxt` by also including significant wave height as another
predicted field.
The model takes along track observations `swh_denoised`, roughly the SWH
equivalent of `sla_filtered`, and makes full-field SWH predictions.
The SWH field is similar to SLA, it takes observation streams, gets encoded to
its own field-specific latent state, then passed to the rest of the network and
predicted full field.

## The Data

Creating the dataset took about 18 minutes.
* The first time I ran this, I ran into the error that prompted PR #37 for nese-dataprep.
* The second time, I noticed the warning
  ```
  [1057 s] [WARNING] [nese.dataprep] /cluster/projects/nn2993k/timith/nese/experimental/all-altimeter/anemoi-hpx6-24h/dataset.zarr: no valid values for 'swh_denoised_jason_1_geodetic'; statistics are NaN
  ```
  and this killed the training, so I removed that data stream
* The stats are a bit funny, with some outliers, see next for a deeper dive

### Min & Max values from the dataset

The minimum values seem to be all over the place, even though the mean
(2.44-2.58 m) and stdev (1.20-1.46m) are consistent.

| Group | Min (m) |
|---|---|
| ERS-1, ERS-1 geodetic, TOPEX, SARAL, SARAL geodetic, Sentinel-3A | −0.235 to −0.001 |
| CryoSat-2, Envisat, ERS-2, Jason-3, Sentinel-3B, TOPEX interleaved | 0.001 to 0.043 |
| Jason-1, Jason-2 and their other phases | 0.37 to 0.43 |

This is apparently due to cross-calibration, which mostly affects calm seas.
In v4, the dataset we pull, CCI uses TOPEX, Jason-1, Jason-2, Jason-3, and Sentinel-6A
as reference missions via their tandem phases.
This is when one flies behind the other on the same orbit.
A buoy based correction is applied, and then every other mission is tied by
crossovers to the reference mission with the longest overlap.

The correction is:
* lookup table between median difference of 0.2m wave-height bins
* linear fit above 2-7m (different per-mission), then 5-point smoothing
* Below the lowest bin, constant offset applied (e.g., this raises Jason-2 min
  from ~0 -> 0.4 m and lowers SARAL 0.46m -> 0.00).
* Outlier test applied iteratively up to 3 times, based on mean +/- 4 stdev of
  surrounding 100km of track, after dropping two most extreme values

Reference: CCI Sea State v4 System Documentation, sections 4.5.4 (Editing) and
4.5.6 (SWH bias correction):
https://cciseastate.gitlab-pages.ifremer.fr/ccidoc/altimetry/baseline.html

## Results

### Version 0

* Compared to `diagnose-uvgxt-v4`, this is the same, except it adds swh as
  noted, and also I re-scaled the uvgxt loss by 0.9 (weights for sla = swh = 1)
  to account for the missing values near the equator
* Nothing done on the dataprep or modeling side to handle the negative values or
  outliers.

Some quick notes on the results:
* Unseen SLA RMSE got 2.4% worse, a bit over 1mm, probably the biggest
  degradation so something to watch
* Bias and unseen MAE degraded too, similar levels
* SLA extrema look good, no change from previous
* SWH beats persistence in unseen RMSE and MAE, and is in fact the lowest
  in the unseen category...
* UVGXT is comparable to persistence in unseen RMSE, but loses to it in MAE.

I don't know why unseen SWH errors are the lowest, that's weird.
But for why they beat persistence, maybe because it's a "rougher" field to
predict than SLA, and SWH prediction gets an advantage from a smoothing effect.
Note that SLA extremes are predicted fairly closely, but SWH and UVGXT extremes
are significantly muted.


#### Model size

~57.4 M trainable parameters, 229.5 MB of weights (fp32)

| Stage | Trainable params | MB |
|---|---:|---:|
| Field encoder (one 2-layer MLP per field, 8 channels each: sla 88, uvgxt 96, swh 88) | **272** | 0.00 |
| Encoder (`GraphTransformerForwardMapper`, no trainable edge attrs) | **3,479,552** | 13.92 |
| Processor (16-layer `RadiusTransformerProcessor`) | **50,413,568** | 201.65 |
| Decoder (`GraphTransformerBackwardMapper`, no trainable edge attrs) | **3,479,042** | 13.92 |
| **Total** | **57,372,434** | **229.49** |

SWH costs 25,177 parameters over `diagnose-uvgxt-v4` (57,347,257): its 8 field
channels x 3 input steps = 24 more input columns into the encoder and decoder
embeddings (12,288 each), its field-encoder MLP (88), and one more output
channel (513). The processor is untouched.

Regarding the field encoder spacing, we have

| Stage | Width per node |
|---|---|
| Field encoder output, per input step | 3 fields x 8 = 24 channels |
| Other inputs, per input step | 12 (`elevation`, `land_fraction`, 7 computed forcings, `mdt`, `mdt_u`, `mdt_v`) |
| Per input step | 36 |
| Encoder input: 3 steps + 4 node-position attributes | 3 x 36 + 4 = **112** |
| Projected to the latent | **512** |
| Decoder output | 2 (`sla`, `swh`); `ugxt`/`vgxt` are then computed from `sla` |


### Version 1

1. Cut negative values from the data, SWH shouldn't be negative
2. Add a positivity constraint to the model for SWH
   (similar to precip in the MLWP anemoi models)

RMSE and MAE vs diagnose-uvgxt-v4
* SLA: slight degradation in unseen category around 2%, but just about 1mm
* UVGXT: <1% differences

Seen vs Unseen Categories
* SLA beats persistence in "seen", but is worse in "unseen"
* UVGXT beats persistence in "seen", but is level with persistence in "unseen"
* SWH beats persistence in all, with best skill in "unseen"

My current take is that the relatively good "unseen" skill in uvgxt and swh is
related to the smoothing effect on these fields.
The model does a better job at capturing the max/min SLA from the data than with
UVGXT and SWH, suggesting that these fields are blurred a bit more.
Hence, they benefit in the RMSE metrics.

Biases:
* SLA: 11-22% degradation, but still <1mm different. This must be noise.
* UVGXT: <1% degradation, ~.5 mm/s

Extremes:
* Negligible changes vs diagnose-uvgxt-v4
* As noted, SWH and UVGXT blur the max/min seemingly significantly

Visually:
* The "observing line" effect noted before is still visible, but I need to
  isolate it a bit better
* The fields "look visually good", except SWH does look a bit blocky.
  Hard to tell if this is checkerboarding, related to overtraining or lack of
  data, or simply due to model resolution.
  Table this for now until we're at higher resolution.

### Version 2

This is the best model from the small training schedule tests done in
`../training-schedule-all-altimeter`.


### Version 2p0

Bump horizontal resolution to hpx8, keep temporal resolution at 24h.
At this resolution, the dataset is 94GB on disk.

See `../throughput-hpx8-06h` for computational throughput tests to determine the
local batch size, num workers, prefetch factor, and flex attention block size.

Scale the peak LR by 1/4 (just a guess) and apply the same factor to
weight decay given the EMA timescale argument by Wang and Aitchison (2025),
given that the model size doesn't really change and dataset size (num. samples)
is the same.

Compared to persistence
* SLA barely beats it at RMSE for seen at t0, and worse at seen at t-48h. Unseen error
  is worse (as with 100km model)
* UVGXT and SWH beat persistence for MAE/RMSE almost all categories, except
  UVGXT MAE unseen. Bias shows similar patterns to 100km model
* SLA max/min are damped relative to obs, more so than 100km model
* UVGXT max/min are damped relative to obs, but UVGXT higher max than 100km
  model whereas SWH somewhat lower...

Compared to v2 (100km model)
* raw errors look better, but it's just a different metric
* relatively worse SLA error in seen locations, but somewhat similar
  seen/unseen behavior

Qualitatively...
* The maps look aesthetically pleasing, but the predictions in data gaps is more
  obvious since there are more grid cells between available input altimeter tracks
* See especially velocity, which is ~0 in the gaps
