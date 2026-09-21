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
`BlockMask`, compiled on CUDA, untested on GPU so far).
Sharded processor edges (multi-GPU model parallelism) are not supported yet.

**Footprint**
`plot_radius_footprint.py` produces processor figures.
As a function of radius at HEALPix level 5:

| radius (km) | Neighbors | % of mesh at 4 passes | % of mesh at 16 passes |
|---|---|---|---|
| 1000 | 70-79 | 8% | 85% |
| 1200 | 295-395 | 12% | 98% |
| 2000 | 103-109 | 33% | 100% |


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

**Laptop training** (20 steps each, same data/seed config; logs in `runs/`):

| processor | train loss (epoch) | val loss (10 batches) | wall time |
|---|---|---|---|
| sliding window ±512, RING | 1.30 | 1.18 | 6:54 |
| 1000 km disk, NEST, `scaled_dot_product_attention` | 1.28 | 1.16 | 7:07 |

The radius processor trains end to end and matches the band at this (tiny) scale.
