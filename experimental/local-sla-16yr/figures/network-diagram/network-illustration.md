# Network illustration — working notes

Handoff for the encoder and decoder architecture figures in this directory.
Covers what was built, the conventions we settled on, what was verified against
the anemoi source, and the threads left open (processor, extra datasets, land
masking).

Companion docs: `README.md` (how to build), `../../context.md` (project scope).

---

## 1. What exists

TikZ figures of the **encoder** (`GraphTransformerForwardMapper`) and the
**decoder** (`GraphTransformerBackwardMapper`), drawn at the block level of
*Attention Is All You Need* Fig. 1, with array shapes on every wire. Landscape,
left-to-right, three lanes (data / edges / latent) converging at the attention
block. The processor is still not drawn.

```
make            # network_diagram.pdf  — 4 pages
make figure     # encoder_standalone.pdf + decoder_standalone.pdf
make png        # rasterise for a quick look
make clean
make CKPT=<other>/inference-last.ckpt      # point at a different run
```

`network_diagram.pdf` is:

| page | |
|---|---|
| 1 | the encoder wire diagram, **symbols only, no numerals** |
| 2 | the decoder wire diagram, same block vocabulary |
| 3 | the notation table, values read from the checkpoint |
| 4 | **Notes** — what this configuration is doing |
| 5 | **Candidates for Architectural Experimentation** — where these depart from GraphCast / plain GNNs / transformers |

The split between the last two pages is the useful one to keep: page 4 states
what is true of this run, page 5 proposes changes to it.

Files: `encoder_figure.tex` and `decoder_figure.tex` are the ones you edit.
`notation.tex` is the single place a symbol is renamed. `dims.tex` and
`symbols_table.tex` are **generated** by `extract_dims.py` — never hand-edit
them.

**`CKPT` now defaults to `runs/v4`,** not the v0 run the first version of these
notes was written against. The Makefile's old default path
(`runs/sea-mask-in-model/...`) no longer exists; that run is now `runs/v0`. v4 is
what `../../training.yaml` describes and it differs from v0 in ways the pages
report: the data mesh is sea-masked (`N_x` 33469, not 49152), there are 35
forcings rather than 28, and every `trainable_size` is 0, so **no parameter is
pinned to the graph any more** — the old "19% of the model" note has been
rewritten accordingly.

## 2. Conventions we settled on

Decided during the session; keep them when extending.

- **Lowercase for variables and functions.** Uppercase is reserved for matrices
  (unused so far), except integer counts, which are always `N_{quantity}`.
- **`ψ` in data space maps to `φ` in latent space.** `φ` (not `h`) for the latent
  state so it never reads as "heads".
- **`N_EE`** for encoder edges, **`N_DE`** for the decoder's. One symbol `a_e`
  serves both edge attribute sets; the counts carry the distinction, and the
  notation caption says the two are separate tensors with separate weights.
- **Source lane on top, destination lane at the bottom, output along the middle
  line** — on both pages. The lanes keep their colours between the pages and swap
  position, so the two figures are structurally identical and the colour tells
  you which space is playing which role.
- **A fourth accent (purple) only for the decoder's output extractor** and the
  output channels. Inactive paths — present in the code, no-ops in this
  configuration — are drawn grey and dashed.
- **Prose belongs in the document, not in a box on the figure.** If something
  needs a paragraph to explain, it is a Notes bullet; the figure gets at most a
  wire label.
- **Lane labels are anchored to their own input box**, not to a fixed offset off
  the lane row, so a box that grows a line cannot hide one.
- **`‖` is concatenation** along the feature axis; the circled **`+` is
  addition**. One meaning per symbol — this is why the table's `⊕`s were changed
  to `‖`.
- **Boxes are labelled by what they are** (`Linear`, `LayerNorm`, `MLP`), never by
  role. The role rides on the wire as a symbol (`k`, `v`, `q`, `r`, `ε`). Six
  Linears in the block: four projections, one for edges, one output projection.
  `MLP` rather than "Feed Forward" because that is anemoi's own name
  (cf. `mlp_hidden_ratio`).
- **No numbers on the figures.** Values live only in the notation table, generated
  from the checkpoint so they cannot drift.

### Layout mechanics

Both figure files place every node off the `\row*` / `\col*` registers at the
top of the file. Reflow by editing those numbers, not by moving nodes. This is
the deliberate departure from `$WORK/nested-eagle-paper/figures/nested-illustration`,
which hard-codes inches because it was pinned to cropped bitmaps.

**Apparent text size is controlled by the aspect ratio, not the font size.** The
figure is set with `\resizebox{\textwidth}{!}`, so only the aspect ratio decides
how much page height it fills. It was 2.5:1 against a 1.5:1 page and got scaled
down until white bands appeared above and below. Spreading the lanes vertically
(now ~1.85:1) fixed both the whitespace and the text size at once. **The lever is
`\rowedge` / `\rowlat`** — pushing lanes apart scales everything up together.

There is a ceiling. The decoder's chain runs ~7 cm further right than the
encoder's (extractor, then the residual add), so its lanes were spread to
compensate and its columns pulled in by ~3 cm. Spread them any further and the
figure plus its caption no longer fit the page: LaTeX ships the caption to a page
of its own and you get a blank page in the middle of the document. Same failure
mode bit the notation table, which is why that page is now two tabulars side by
side rather than one.

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

## 4. Verified decoder forward pass

Same checkout, same single-dataset case. `x_src` is the processor output with the
latent skip already added; `x_dst` is the **raw** `ψ` the encoder handed back.

```
  x_src = x_latent_proc                          (N_z, N_φ)
  x_dst = ψ                                      (N_x, N_ψ)   raw, from the encoder
  a_e, edge_index from StaticGraphProvider       (N_DE, N_ae), (2, N_DE)  KNN k=3

GraphTransformerBackwardMapper:
  pre_process:  x_src UNCHANGED                  # no emb_nodes_src exists
                Linear (emb_nodes_dst) : (N_x, N_ψ) → (N_x, N_φ)

  GraphTransformerMapperBlock:   # identical to the encoder's, roles swapped
    s0 = latent, d0 = data
    r  = lin_self(LayerNorm_dst(d0))   → (N_x, N_φ)
    q  = lin_query(LayerNorm_dst(d0))  → (N_x, N_H, N_c)
    k, v = lin_key/value(LayerNorm_src(s0)) → (N_z, N_H, N_c)
    ε  = lin_edge(a_e)                 → (N_DE, N_H, N_c)
    α_ij = softmax_{i∈𝒩(x_j)} ...       # over exactly 3 latent nodes
    out = projection(out + r); out = out + d0; out = MLP(LayerNorm(out)) + out

  post_process:  node_data_extractor = Sequential(LayerNorm, Linear)
                 (N_x, N_φ) → (N_x, N_y)

then in AnemoiModelEncProcDec._assemble_output:
  rearrange → (batch, time, ensemble, grid, vars)
  x_out[..., _internal_output_idx] += x_skip[..., _internal_input_idx]
                                 ↑ prognostic only — EMPTY here
  boundings (none configured)
  → sla at t+Δt                                  (N_x, 1)
```

Three things this pinned down:

1. **The decoder has no source embedding at all.** `pre_process` embeds only the
   destination (`mapper.py:698-701`). Harmless while `num_channels` is the width
   on both sides, but the latent stream gets no per-mapper reprojection on the
   way out.
2. **`node_data_extractor` is built from `nn.LayerNorm` / `nn.Linear` directly,
   not from `layer_kernels`.** A change to `model.layer_kernels` silently misses
   the one layer that produces the model's output.
3. **The residual add is dead in this configuration.** `_internal_output_idx` is
   `model.output.prognostic`, and with `sla` declared under `diagnostic:` and
   nothing prognostic, that index set is empty — verified on the checkpoint, both
   `_internal_input_idx` and `_internal_output_idx` are empty tensors.
   `SkipConnection` still runs; its result is added to no column. The config
   comment in `training.yaml` says this; it is now checked rather than asserted,
   and `extract_dims.py` will fail loudly if `N_y != |P| + |D|` ever stops
   holding.

### Where the diagnostic "comes in"

Nowhere, in the sense a diagram could show. There is no separate diagnostic
branch. The distinction lives entirely in the **index sets**: `model.output` is
`prognostic ∪ diagnostic`, the extractor's Linear emits all of them at once in
one matrix, and the only thing separating the two is which columns
`_assemble_output` adds the input state back onto.

So the decoder page does not try to draw it. It shows the extractor Linear
(accented) feeding a residual `⊕` whose other input is greyed and dashed and
labelled `added on P only` — and the explanation lives in the **Notes** section
on page 4, as prose. An explanatory box floating over the diagram was tried first
and removed: it implied there was a mechanism there to point at.

## 5. Numbers from this checkpoint

`runs/v4/checkpoint/23c396bfc6ff4049a776ee6aca7af43f/inference-last.ckpt`

| | | |
|---|---|---|
| `N_x` | data (ocean) nodes | 33469 (HEALPix res 6, sea-masked) |
| `N_z` | latent (hidden) nodes | 12288 (HEALPix res 5) |
| `N_EE` | encoder edges | 107950 (CutOff 200 km) |
| `N_DE` | decoder edges | 100407 (KNN k=3, = 3·N_x) |
| `N_u` / `N_T` | features per step / input steps | 35 / 3 |
| `N_ax`, `N_az` | node attrs (sin/cos lat-lon only) | 4, 4 |
| `N_ψ` | assembled input | 109 = 3·35 + 4 |
| `N_ae` | edge attrs (length, direction) | 3 |
| `N_φ` / `N_H` / `N_c` | latent channels / heads / per head | 128 / 4 / 32 |
| `N_f` | MLP hidden | 512 |
| `N_y` | output channels | 1 = 0 prognostic + 1 diagnostic (`sla`) |

Graph degree: in-degree per latent node 0–15, **mean 8.8**; **2967 of 12288**
latent nodes have empty `𝒩(z_j)` (land masking). On the way out every one of the
33469 data nodes has exactly 3 incoming edges — KNN, not a cutoff.

Parameters: encoder 230272, processor 791552, decoder 230017, **total 1251841**.
Trainable node and edge attributes: **0** — every `trainable_size` in
`training.yaml` is 0. (In v0 they were 338780, 19% of the model; if they are ever
turned back on, that note on page 4 is written to pick the numbers back up.)

## 6. Oddities (page 5 of the PDF has the full list)

The ones worth chasing:

- **Aggregation is degree-normalised; GraphCast's is not.** Softmax → unit-weight
  average regardless of degree; GraphCast sums MLP messages, so its aggregate
  grows with degree. For a DA-like problem observation density *is* much of the
  signal, and the attention output carries no magnitude cue for it. The
  `obs_count_*` channels reach the values via `ψ`, but indirectly — unverified
  whether the model exploits it.
- **The decoder's receptive field is three latent nodes, fixed.** KNN k=3, so the
  softmax normalises over three logits regardless of local latent density, and a
  data node in a sparsely observed region is reconstructed from exactly as many
  latent nodes as one in a dense region. The encoder gathers from 0–15. Nothing
  in training can widen either — fix it in `anemoi-graphs`.
- **Nothing is pinned to the graph in v4** (`trainable_size: 0` everywhere), which
  is a departure from stock anemoi. Turning it back on adds order `N_x + N_z`
  parameters per node column and `N_EE + N_DE` per edge column, zero-initialised,
  transferable to no other graph or resolution. First suspect if spatial
  generalisation looks surprisingly good after that change.

Also on the list: no edge updates (GraphCast is a full interaction network);
ε enters both key and value (inherited from PyG `TransformerConv`, not an anemoi
invention); no self-loop in the softmax so the own-state/neighbour balance is
unbounded; the decoder's destination input is raw `ψ` so the input is embedded
twice by two independent matrices (candidate ablation); the decoder has no source
embedding; `node_data_extractor` bypasses `layer_kernels`; the model has no
persistence path because every output channel is diagnostic; latents summed with
no gate.

## 7. Multi-dataset findings (for the atmosphere/forcings plan)

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

## 8. Open threads

- **Processor figure.** `TransformerProcessor`, 4 layers, sliding window 512 over
  the 12288 latent nodes. All model depth lives here.
- Reuse `notation.tex` + `dims.tex` and add `processor_figure.tex` beside the
  other two; extend `ROWS` in `extract_dims.py` for any new dimensions. A row
  added after the `SPLIT` marker lands in the notation table's second column.
- **Other variables / datasets** — see §7.
- **Land masking** — deliberately parked. The 2967 empty-neighbourhood latent
  nodes and `sea_mask` interact with loss masking; revisit if a land-inclusive
  dataset is added.
