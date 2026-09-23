# Non-Banded Processor

Currently, I am using a sliding window transformer processor, which acts as
follows.
The latent mesh nodes are ring ordered, so that we can efficiently grab "chunks"
of the array that correspond to zonal bands.
By setting a large enough attention window, our zonal bands extend far enough
north and south to get a good enough local stencil to capture physically
relevant processes for a given query node.
By iteratively evaluating the processor, we get a large enough radius of
influence to capture relevant dynamics.
A visualization of this processor can be found at
`../local-sla-02yr/graphs/figures/sliding-window/rmData.*`.

**Notes**:
* Right now I have yamls associated with a laptop setup, corresponding
  to just 4 processor steps, but we will use 16 when this is actually implemented.
* In that figures directory, `rmDataHidden.*` correspond to experiments where I
  was removing hidden nodes from the graph. I don't need want to do that, in order
  to prepare for coupled earth system modeling, so we can ignore it.

The processor is extremely efficient because we can grab contiguous chunks of
arrays, and feed it into flash-attention, which is well-engineered.
This processor worked very well for me in the past, when I was working on ML
Weather Prediction, since meteorological quantities propagate more zonally than
meridionally (to first order).
However, I would prefer to use a radius based approach for ocean emulation,
because:
* ocean quantities are more generally confined by bathymetry than zonal bands,
    * although, of course there are exceptions, like in the tropics which is highly
      zonally constricted
* ocean quantities have very different properties per-basin, so it would not
  make sense to keep track of the opposite end of the globe, if we can avoid it


The issue is efficiency.
It is probably not very computationally efficient to draw a radius and gather
effected points for every query node on the latent mesh.
The GenCast processor (https://www.nature.com/articles/s41586-024-08252-9)
(and all subsequent WeatherNext models, e.g. WN2 or FGN http://arxiv.org/abs/2506.10772)
use something that looks radius based, but apparently achieves this due to a
node ordering trick (see code base at
https://github.com/google-deepmind/weathernext).

Could we use an idea like this to get a processor that acts on, e.g. 1000km
radii (just an example, this is something I would want to tune) around a query
node?

Note that we have already (by accident) seen what happens when we simply order
the latent mesh nodes in a nested HEALPix ordering.
See figures at `../local-sla-02yr/graphs/figures/sliding-window-nest`.
In those figures we see that the pixels in the lower HEALPix levels light up.
E.g. For a query node at HEALPix level 5, we see the all the nodes inside of
levels 2-4 (or something like that), almost like a radius.
The problem with using this is that the radius of influence is not equal for
nodes that sit toward the center versus the edges of these lower levels, and I
need the stencil to be nearly symmetric, similar to the sliding window
transformer (with some exceptions, for example symmetry won't be achieved at the
poles).

## Findings

**How GenCast / WeatherNext do it.**
In `weathernext/utils/sparse_transformer.py` the attention mask is the k-hop
neighbourhood of the mesh (`adj_mat ** attention_k_hop`). The node ordering
is a Reverse Cuthill–McKee permutation (`icosahedral_mesh.get_permutation_to_banded`),
which does not define the stencil; it only pulls the mask's nonzeros close to
the diagonal so that a block-sparse kernel (TPU splash attention) can skip
empty tiles.
So the "trick" is: explicit graph mask for the stencil + an ordering that makes
that mask block-sparse.

**What we do.**
`nese_training.radius_attention.RadiusTransformerProcessor` is anemoi's
`TransformerProcessor` with its attention replaced by a mask built from a
`(hidden, to, hidden)` `CutOffEdges` edge set (`cutoff_distance_km: 1000` in
`training.yaml`). Because the stencil is set by distance, it is symmetric
regardless of node order, which removes the nested-HEALPix asymmetry.
The hidden mesh is switched to nested ordering because that keeps each 128-node
block spatially compact, which matters for block sparsity.
`attention_implementation`: `scaled_dot_product_attention` (dense mask, laptop) or `flex_attention` (with a
`BlockMask`, compiled on CUDA, runs on GH200).
Sharded processor edges (multi-GPU model parallelism) are not supported yet.

**Footprint**
`plot_radius_footprint.py` produces processor figures.
As a function of radius at HEALPix level 5:

| radius (km) | Neighbors | % of mesh at 4 passes | % of mesh at 16 passes |
|---|---|---|---|
| 1000 | 70-79 | 8% | 85% |
| 1200 | 103-113 | 12% | 98% |
| 2000 | 293-308 | 33% | 100% |

For comparison, the sliding window (RING order, ±W tokens, no wraparound) reaches
±(passes × W) tokens, so its coverage depends on latitude: it is smallest at the
poles, where the index range gets clipped.
Values are the mean over all latent nodes (minimum in parentheses):

| window W | keys per query | % of mesh at 4 passes | % of mesh at 16 passes |
|---|---|---|---|
| 128 | 257 | 8% (4%) | 31% (17%) |
| 256 | 513 | 16% (8%) | 56% (33%) |
| 512 (current) | 1025 | 31% (17%) | 89% (67%) |
| 1024 | 2049 | 56% (33%) | 100% |

At 16 passes, the 1200 km radius covers more of the mesh (98%) than the ±512
window (89%), with ~10× fewer keys per query in its exact stencil.


**Cost at 1000km Radius**, query-key pairs per head per layer (no timings):

| layout | keys per query | % of dense |
|---|---|---|
| sliding window ±512, RING (current) | 1004 | 8.2 |
| exact 1000 km disk | 74 | 0.6 |
| block-sparse 128, NEST order | 1099 | 8.9 |
| block-sparse 128, RING order | 1744 | 14.2 |

With nested ordering and flex attention, the disk costs about the same as the
current band, but the stencil is local and isotropic. Ring order would cost ~1.6×
more, because a 128-node ring block spans a whole latitude circle.
Whether flex attention runs as fast as flash attention per computed
pair is the open question for Olivia.

### Results

Error metrics show little change from the baseline.
The RMSE shows a tiny degradation, and bias shows something similar.
However, this should probably not be taken seriously given the results in
`../olivia-sla-baseline`, which show similar order-of-magnitude differences
between two models with the same specifications, meaning that the differences
can just come from stochasticity of the training procedure itself.

Comparing the loss curve between the baseline and radius-processor seems to
indicate that the latter overfits a bit more, which could account for these
(albeit) tiny degradations.
The radius-processor validation loss initializes noticeably lower than with the
sliding window.
By eyeball, it looks like the radius-processor validation loss flattens out
around 6000 iterations, whereas this happens around 8000 iterations in the
sliding window processor.

```
Radius-processor relative to Baseline, pooled over groups
---------------------------------------------------------
(time mean of |metric|; negative is an improvement)

                            RMSE                       MAE                      Bias
Unseen    0.06357 ->  0.06386 (  +0.5%)    0.03815 ->  0.03806 (  -0.2%)    0.00207 ->  0.00223 (  +7.6%)
+0h       0.04181 ->  0.04191 (  +0.2%)    0.02609 ->  0.02609 (  +0.0%)    0.00054 ->  0.00057 (  +5.8%)
-24h      0.04287 ->  0.04297 (  +0.2%)    0.02672 ->  0.02673 (  +0.0%)    0.00065 ->  0.00066 (  +0.5%)
-48h      0.04300 ->  0.04311 (  +0.3%)    0.02692 ->  0.02692 (  +0.0%)    0.00065 ->  0.00067 (  +2.4%)
```

### Computational Efficiency

The radius processor (1200 km, flex attention) trains ~14% slower than the sliding window,
same 10k steps on 4 GH200s:

| run | train it/s (epoch 17) | `Trainer.fit` time | job wallclock | GPU hours |
|---|---|---|---|---|
| baseline v0 (2225392) | 5.04 | 34.2 min | 35.3 min | 2.4 |
| baseline v1 (2307098) | 5.02 | 34.4 min | 35.5 min | 2.4 |
| radius (2305595) | 4.42 | 39.2 min | 40.5 min | 2.7 |

Baseline run-to-run spread is <1%, so the slowdown is real.

**Block size at 1200 km** (NEST order), query-key pairs per head per layer (no timings):

| Q block × KV block | nonzero tiles | keys per query | × exact disk | % of dense |
|---|---|---|---|---|
| 32 × 32 | 5544 | 462 | 4.4 | 3.8 |
| 64 × 64 | 1704 | 568 | 5.4 | 4.6 |
| 128 × 64 | 1128 | 752 | 7.1 | 6.1 |
| 128 × 128 (current) | 824 | 1099 | 10.4 | 8.9 |
| 256 × 128 | 552 | 1472 | 13.9 | 12.0 |

The exact disk has 106 keys per query.
Not a single tile is full at any block size, so flex attention evaluates
`mask_mod` (a gather from the dense N×N mask) in every tile it computes.
At 128 × 128, the radius processor computes about as many pairs as the ±512
sliding window (1004), so the 14% slowdown points to a higher cost per pair
(the `mask_mod` gathers, a less tuned kernel than flash attention) rather
than to more pairs.
With `num_channels: 512` and 16 heads, the head dimension is 32. For that size,
torch 2.10 picks 64 × 64 kernel tiles on Hopper for both the forward and
backward passes, so the kernel already splits each 128 × 128 mask block into
four 64 × 64 tiles.
A 64 block mask therefore cuts 48% of the computed pairs while the kernel
keeps its tile size, and it is the natural first try.
128 × 64 is a milder option (−32%).
The block mask size is set by `block_size` on the processor (`training.yaml`),
and it must be a multiple of the kernel tile, otherwise flex attention raises an error.

### TODO

* benchmark `block_size: 64` (and `[128, 64]`) against 128 on Olivia
