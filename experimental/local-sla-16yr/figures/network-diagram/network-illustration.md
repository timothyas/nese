# Network illustration — working notes

Handoff for the encoder architecture figure in this directory. Covers what was
built, the conventions we settled on, what was verified against the anemoi
source, and the threads left open (processor, decoder, extra datasets, land
masking).

Companion docs: `README.md` (how to build), `../../context.md` (project scope).

---

## 1. What exists

A TikZ figure of the **encoder only** — `GraphTransformerForwardMapper` — drawn
at the block level of *Attention Is All You Need* Fig. 1, with array shapes on
every wire. Landscape, left-to-right, three lanes (data / edges / latent)
converging at the attention block.

```
make            # network_diagram.pdf  — 3 pages
make figure     # encoder_standalone.pdf, tight-cropped for \includegraphics
make png        # rasterise for a quick look
make clean
make CKPT=<other>/inference-last.ckpt      # point at a different run
```

`network_diagram.pdf` is:

| page | |
|---|---|
| 1 | the encoder wire diagram, **symbols only, no numerals** |
| 2 | the notation table, values read from the checkpoint |
| 3 | **Notes** — where this departs from GraphCast / plain GNNs / transformers |

Files: `encoder_figure.tex` is the one you edit. `notation.tex` is the single
place a symbol is renamed. `dims.tex` and `symbols_table.tex` are **generated**
by `extract_dims.py` — never hand-edit them.

## 2. Conventions we settled on

Decided during the session; keep them when extending.

- **Lowercase for variables and functions.** Uppercase is reserved for matrices
  (unused so far), except integer counts, which are always `N_{quantity}`.
- **`ψ` in data space maps to `φ` in latent space.** `φ` (not `h`) for the latent
  state so it never reads as "heads".
- **`N_EE`** for encoder edges — leaves **`N_DE`** free for the decoder's.
- **`‖` is concatenation** along the feature axis; the circled **`+` is
  addition**. One meaning per symbol — this is why the table's `⊕`s were changed
  to `‖`.
- **Boxes are labelled by what they are** (`Linear`, `LayerNorm`, `MLP`), never by
  role. The role rides on the wire as a symbol (`k`, `v`, `q`, `r`, `ε`). Six
  Linears in the block: four projections, one for edges, one output projection.
  `MLP` rather than "Feed Forward" because that is anemoi's own name
  (cf. `mlp_hidden_ratio`).
- **No numbers on the figure.** Values live only in the page-2 table, generated
  from the checkpoint so they cannot drift.

### Layout mechanics

`encoder_figure.tex` places every node off the `\row*` / `\col*` registers at the
top of the file. Reflow by editing those numbers, not by moving nodes. This is
the deliberate departure from `$WORK/nested-eagle-paper/figures/nested-illustration`,
which hard-codes inches because it was pinned to cropped bitmaps.

**Apparent text size is controlled by the aspect ratio, not the font size.** The
figure is set with `\resizebox{\textwidth}{!}`, so only the aspect ratio decides
how much page height it fills. It was 2.5:1 against a 1.5:1 page and got scaled
down until white bands appeared above and below. Spreading the lanes vertically
(now ~1.85:1) fixed both the whitespace and the text size at once. **The lever is
`\rowedge` / `\rowlat`** — pushing lanes apart scales everything up together.

## 3. Verified encoder forward pass

Traced through `/Users/timith/work/anemoi-core/models/src/anemoi/models/`. Note
that checkout is **post-refactor**: mappers live in a single `layers/mapper.py`
(no package dir, no mixins), `shard_shapes` is now `BipartiteGraphShardInfo`, and
the model is multi-dataset (`x` is a `dict[str, Tensor]`).

Single dataset `ocean`, non-distributed, batch 1, ensemble 1:

```
input                                       (N_T, N_x, N_u)
  ├─ residual  ────────────────────────────────────► bypasses enc/proc/dec
  └─ rearrange "b t e g v -> (b e g) (t v)"    → (N_x, N_T·N_u)
       ‖ a_x  from NamedNodesAttributes("ocean")    (N_x, N_ax)
       = ψ                                          (N_x, N_ψ)
                                                     │
                                                     └──► raw, UNCHANGED, is the
                                                          decoder's dst input

  a_z  from NamedNodesAttributes("hidden")     (N_z, N_az)
  a_e, edge_index from StaticGraphProvider     (N_EE, N_ae), (2, N_EE)  [dst-sorted]

GraphTransformerForwardMapper:
  Linear (emb_nodes_src) : (N_x, N_ψ)  → (N_x, N_φ)
  Linear (emb_nodes_dst) : (N_z, N_az) → (N_z, N_φ)

  GraphTransformerMapperBlock:
    s0, d0 = the two embedded streams               # saved skips
    sN = LayerNorm_src(s0)     ;  dN = LayerNorm_dst(d0)
    r  = lin_self(dN)          → (N_z, N_φ)         # NOT reshaped into heads
    q  = lin_query(dN)         → (N_z, N_H, N_c)
    k  = lin_key(sN)           → (N_x, N_H, N_c)
    v  = lin_value(sN)         → (N_x, N_H, N_c)
    ε  = lin_edge(a_e)         → (N_EE, N_H, N_c)   [edge_pre_mlp = Identity]

    GraphTransformerConv  (src = edge_index[0], dst = edge_index[1]):
      α_ij = softmax_{i∈𝒩(j)} q_j·(k_i + ε_ij)/√N_c        (N_EE, N_H)
      out_j = Σ_{i∈𝒩(j)} α_ij (v_i + ε_ij)                 (N_z, N_H, N_c)

    out = projection(out + r)      # RESIDUAL 1 — inside, pre-projection
    out = out + d0                 # RESIDUAL 2 — pre-norm dst skip
    φ   = MLP(LayerNorm(out)) + out    # RESIDUAL 3
    s1  = s0                       # src is a straight wire (update_src_nodes=False)

  returns (ψ_raw, φ)               # φ : (N_z, N_φ) → processor
```

Things that were fuzzy before tracing, and that the figure now gets right:

1. Three residual adds on the latent path, **zero** on the data path.
2. Residual 1 is not a skip — `r` is added *before* `projection`, so it passes
   through `W^O` rather than around it.
3. The LayerNorm does **not** produce k/v or q/r pairs. It emits one tensor; the
   fork after it is a fan-out read by two independent Linears. Anemoi does *not*
   use a fused `Linear(N_φ → 3N_φ)` + chunk (`block.py:550-553`).
4. `r` is never reshaped into heads; it waits for the attention output to be
   rearranged back to `(N_z, N_φ)`.
5. The mapper returns the **raw, un-embedded** `ψ` (`mapper.py:597`).
   `emb_nodes_src` runs on a chunk-local copy inside the chunk loop.
6. `num_chunks=4` partitions destination nodes; edges are dst-sorted, so it is
   exact — memory optimisation only, deliberately not drawn.

### `𝒩(j)` and what the softmax normalises

`𝒩(j)` = data nodes with an edge into latent node `j`. `Σ_{i∈𝒩(j)} α_ij = 1` per
head, so the weights are relative *within a patch* and there is no competition
between different latent nodes. **The receptive field is a graph decision, not a
learned one** — training re-weights inside `𝒩(j)` but can never attend outside
it. Fix it in `anemoi-graphs`, not in training.

## 4. Numbers from this checkpoint

`runs/sea-mask-in-model/checkpoint/7a74d146c6cf4206a51f2217fad5bdeb/inference-last.ckpt`

| | | |
|---|---|---|
| `N_x` | data (ocean) nodes | 49152 (HEALPix res 6) |
| `N_z` | latent (hidden) nodes | 12288 (HEALPix res 5) |
| `N_EE` | encoder edges | 107950 (CutOff 200 km, sea-masked) |
| `N_u` / `N_T` | features per step / input steps | 28 / 3 |
| `N_ax`, `N_az` | node attrs (4 sin/cos latlon + 2 trainable) | 6, 6 |
| `N_ψ` | assembled input | 90 = 3·28 + 6 |
| `N_ae` | edge attrs (3 static + 2 trainable) | 5 |
| `N_φ` / `N_H` / `N_c` | latent channels / heads / per head | 128 / 4 / 32 |
| `N_f` | MLP hidden | 512 |

Graph degree: in-degree per latent node 0–15, **mean 8.8**; **2967 of 12288**
latent nodes have empty `𝒩(j)`; 33469 of 49152 data nodes feed at least one
latent node. (By design — land masking.)

Parameters: 338780 trainable node+edge attributes vs 228352 in the whole encoder
module, 1787339 in the model → **19% of the model is pinned to specific graph
elements**.

## 5. Oddities (page 3 of the PDF has the full list)

The two worth chasing:

- **Aggregation is degree-normalised; GraphCast's is not.** Softmax → unit-weight
  average regardless of degree; GraphCast sums MLP messages, so its aggregate
  grows with degree. For a DA-like problem observation density *is* much of the
  signal, and the attention output carries no magnitude cue for it. The
  `obs_count_*` channels reach the values via `ψ`, but indirectly — unverified
  whether the model exploits it.
- **19% of parameters pinned to the graph.** Zero-initialised, can memorise
  per-location behaviour, transfer to no other graph or resolution. First suspect
  if spatial generalisation disappoints.

Also on the list: no edge updates (GraphCast is a full interaction network);
ε enters both key and value (inherited from PyG `TransformerConv`, not an anemoi
invention); no self-loop in the softmax so the own-state/neighbour balance is
unbounded; the decoder's skip carries raw `ψ` so the input is embedded twice by
two independent matrices (candidate ablation); latents summed with no gate.

## 6. Multi-dataset findings (for the atmosphere/forcings plan)

Verified in `models/encoder_processor_decoder.py`:

- **Every dataset gets an encoder AND a decoder, unconditionally** (`:41`, `:79`,
  `:300`). No encoder-only dataset in this code path. Each needs *both* graph
  edge sets — `(ds,"to","hidden")` and `("hidden","to",ds)` — or construction
  KeyErrors.
- Latents are **summed**, not concatenated or gated: `x_latent =
  sum(dataset_latents.values())` (`:273`). No learned weighting between datasets.
- `x_hidden_latent` is built **once** (`:229`) and handed to every encoder, so
  with K datasets a latent node receives K attribute-derived residual terms.
- `model.encoder` is one config instantiated per dataset (`:51`) — separate
  weights, shared hyperparameters. Unverified whether per-dataset overrides exist.

Guidance that follows:

| what | where | why |
|---|---|---|
| sin/cos julian day, insolation | extra channels in `ocean` under `forcing:` | same node set; costs one wider Linear |
| sin/cos lat-lon | **already in `a_x`** — don't add | `graph.py:95-98` stores `cat([sin,cos])` |
| winds / land-defined variables | own dataset | genuinely different node set |

A separate dataset on the *same* grid would duplicate all 107950 edges and add a
second attention block. It also changes the semantics: a forcing in its own
dataset arrives spatially-attended over its own neighbourhood, decoupled from the
SLA at the same location; concatenated into `ψ` it travels through the same
attention weight as its node's own SLA.

## 7. Open threads

- **Processor figure.** `TransformerProcessor`, 4 layers, sliding window 512 over
  the 12288 latent nodes. All model depth lives here.
- **Decoder figure.** `GraphTransformerBackwardMapper` — has no `emb_nodes_src`,
  and adds `node_data_extractor = Sequential(LayerNorm, Linear)` as
  `post_process` (`mapper.py:688-705`). KNN edges, k=3.
- Reuse `notation.tex` + `dims.tex` and add `processor_figure.tex` /
  `decoder_figure.tex` beside `encoder_figure.tex`; extend `ROWS` in
  `extract_dims.py` for any new dimensions. `N_DE` is already reserved.
- **Other variables / datasets** — see §6.
- **Land masking** — deliberately parked. The 2967 empty-neighbourhood latent
  nodes and `sea_mask` interact with loss masking; revisit if a land-inclusive
  dataset is added.
