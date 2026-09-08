# Network diagrams

TikZ figures of the encoder (`GraphTransformerForwardMapper`), processor
(`TransformerProcessor`) and decoder (`GraphTransformerBackwardMapper`) of the
`AnemoiModelEncProcDec` trained here, drawn at the block level of *Attention Is
All You Need* Fig. 1 with every wire annotated by its array shape.

    make            # network_diagram.pdf
    make figure     # {encoder,processor,decoder}_standalone.pdf
    make png        # rasterise for a quick look
    make clean
    make CKPT=<other>/inference-last.ckpt      # a different run

`CKPT` defaults to `runs/v4`, the run `../../training.yaml` describes. Other runs
differ in ways the pages report — `v0` has trainable node/edge attributes and an
unmasked data mesh — so all wording that depends on the checkpoint is generated
from it, not typed in, and follows whichever one you build against.

| page | |
|---|---|
| 1 | encoder wire diagram, **symbols only, no numerals** |
| 2 | processor wire diagram |
| 3 | decoder wire diagram, same block vocabulary |
| 4 | notation table, values read from the checkpoint |
| 5 | **Notes** — what this configuration is doing |
| 6 | **Candidates for Architectural Experimentation** — where these depart from GraphCast / plain GNNs / transformers |

Pages 5 and 6 split on a useful line: page 5 states what is true of this run,
page 6 proposes changes to it.

| file | |
|---|---|
| `encoder_figure.tex`, `processor_figure.tex`, `decoder_figure.tex` | the `tikzpicture`s — the files you edit |
| `notation.tex` | symbol macros; the one place a symbol is renamed |
| `network_diagram.tex` | the six-page landscape wrapper |
| `*_standalone.tex` | one figure alone, for `\includegraphics` |
| `extract_dims.py` | reads the dimensions out of a checkpoint |
| `dims.tex`, `symbols_table.tex` | **generated** — never hand-edit, `make` overwrites them |

## Conventions

- **Lowercase for variables and functions**; uppercase reserved for matrices,
  except integer counts, which are always `N_{quantity}` — including set sizes
  (`N_P`, not `|P|`).
- **`ψ` in data space maps to `φ` in latent space.** `φ` not `h`, so it never
  reads as "heads". `N_EE` / `N_DE` for encoder / decoder edges; one symbol `a_e`
  serves both attribute sets and the counts carry the distinction.
- **`‖` is concatenation** along the feature axis, circled **`+` is addition**.
  One meaning per symbol.
- **Boxes are labelled by what they are** (`Linear`, `LayerNorm`, `MLP`), never by
  role; the role rides on the wire as a symbol (`k`, `v`, `q`, `r`, `ε`). `MLP`
  because that is anemoi's own name (cf. `mlp_hidden_ratio`).
- **Source lane on top** (supplies `k`, `v`), edges in the middle, **destination
  lane at the bottom** (supplies `q`, `r`), destination stream leaving along the
  middle line — on the two mapper pages. Lanes keep their colours and swap
  position, so the colour tells you which space is playing which role. The
  processor has one space and no edges, so it is a single lane in the latent
  accent. A fourth accent marks only the decoder's output extractor and output
  channels; paths that are inert in this configuration are grey and dashed.
- **Repetition is a dashed box, not repeated boxes.** The processor's `×N_L` is
  drawn once and fitted with a `\node[fit=...]`, which needs the tikz `fit`
  library and named coordinates — a bare `(\colF, \rowattnskip)` in a `fit=` list
  is a pgf error, so route points that the box must enclose get `\coordinate`
  names.
- **No numbers on the figures.** Values live only in the notation table.
- **Prose belongs in the document, not in a box on the figure.** If it needs a
  paragraph, it is a Notes bullet; the figure gets at most a wire label.

### Layout mechanics

Every node is placed off the `\row*` / `\col*` registers at the top of each figure
file — reflow by editing those numbers, not by moving nodes. (Deliberately unlike
`$WORK/nested-eagle-paper/figures/nested-illustration`, which hard-codes inches
because it was pinned to cropped bitmaps.) Lane labels anchor to their own input
box, so a box that grows a line cannot hide one.

**Apparent text size is set by the aspect ratio, not the font size** — the figures
are placed with `\resizebox{\textwidth}{!}`. Pushing lanes apart scales everything
up together. There is a ceiling: spread them too far and the figure plus its
caption no longer fit, and LaTeX ships the caption to a page of its own, leaving a
blank page mid-document. The same failure mode is why the notation table is two
tabulars side by side rather than one.

## Generated numbers

`extract_dims.py` reads every value off the live modules in the checkpoint, so
they cannot drift from the trained model. It asserts `N_psi == N_T*N_u + N_ax`,
`N_phi == N_H*N_c`, `N_y == N_P + N_D`, that the decoder's extractor still emits
the model's output width, and that the backward mapper still has no source
embedding — so a config or upstream change that breaks what the figures assume
fails loudly instead of drawing a lie. It also derives the incoming-degree
distribution of both graphs and whether the attributes carry a trainable tail
(`\valtrainablenote` carries the wording for either case).

It also measures the processor's attention window in degrees of latitude, since
`window_size` is a half-width on the node *index* and only the graph knows what
that is worth on the sphere, and asserts the hidden nodes are still in ring order
(the measurement is meaningless under nested ordering).

From `runs/v4`: `N_x` 33469 (HEALPix res 6, sea-masked), `N_z` 12288 (res 5),
`N_EE` 107950 (cutoff 200 km), `N_DE` 100407 (KNN k=3, = 3·N_x), `N_φ` 128,
`N_L` 4, `N_w` 512, `N_y` 1 = 0 prognostic + 1 diagnostic (`sla`). Encoder
in-degree 0–15, mean 8.8, empty for 2967 of 12288 latent nodes (land masking);
decoder in-degree exactly 3 everywhere; processor window ≈ 11° of latitude and
every longitude. Parameters: encoder 230272, processor 791552, decoder 230017,
total 1251841, with **nothing** pinned to the graph (every `trainable_size` is 0;
in v0 it was 338780, 19% of the model).

## Verified encoder forward pass

Traced through `$WORK/anemoi-core/models/src/anemoi/models/`. That checkout is
**post-refactor**: mappers live in a single `layers/mapper.py` (no package dir, no
mixins), `shard_shapes` is now `BipartiteGraphShardInfo`, and the model is
multi-dataset (`x` is a `dict[str, Tensor]`). Single dataset `ocean`,
non-distributed, batch 1, ensemble 1:

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

Points the trace settled, and the figure now gets right:

1. Three residual adds on the latent path, **zero** on the data path.
2. Residual 1 is not a skip — `r` is added *before* `projection`, so it passes
   through `W^O` rather than around it.
3. The LayerNorm does **not** produce k/v or q/r pairs. It emits one tensor and
   the fork after it is a fan-out read by two independent Linears; anemoi does
   *not* use a fused `Linear(N_φ → 3N_φ)` + chunk (`block.py:550-553`).
4. `r` is never reshaped into heads; it waits for the attention output to be
   rearranged back to `(N_z, N_φ)`.
5. The mapper returns the **raw, un-embedded** `ψ` (`mapper.py:597`).
   `emb_nodes_src` runs on a chunk-local copy inside the chunk loop.
6. `num_chunks=4` partitions destination nodes; edges are dst-sorted so it is
   exact — memory optimisation only, deliberately not drawn.

`𝒩(j)` is the set of data nodes with an edge into latent node `j`, and
`Σ_{i∈𝒩(j)} α_ij = 1` per head — so weights are relative *within a patch*, with no
competition between latent nodes. **The receptive field is a graph decision, not a
learned one**: training re-weights inside `𝒩(j)` but can never attend outside it.
Fix it in `anemoi-graphs`, not in training.

## Verified processor forward pass

Between the two mappers, and the only stage that is not a graph operation:
`processor_graph_provider` is a `NoOpGraphProvider`, and `TransformerProcessor`
takes `edge_attr` / `edge_index` only to drop them (`processor.py:298-316`). The
hidden→hidden graph is never used — and `training.yaml` never defines one.

```
  x_latent = Σ over datasets of the encoder outputs      (N_z, N_φ)

TransformerProcessor:  N_L blocks, separate weights, run in
  num_layers // num_chunks-sized groups under gradient checkpointing

  TransformerProcessorBlock (block.py:174-196), standard pre-norm:
    x = x + attention(LayerNorm_attn(x))     # add AFTER the output projection
    x = x + mlp(LayerNorm_mlp(x))

  MultiHeadSelfAttention (attention.py):
    q = lin_q(x); k = lin_k(x); v = lin_v(x)   # three separate Linears, bias=False
    α_ij = softmax_{|i-j| ≤ N_w} (q_i·k_j / √N_c)
    out  = projection(Σ_j α_ij v_j)            # projection has bias=True

then in AnemoiModelEncProcDec.forward:
  if latent_skip:  x_latent_proc = processor(x_latent) + x_latent
  → φ' to every decoder                                 (N_z, N_φ)
```

1. **The window is over the node index, not the graph** — an SDPA band mask on
   `|i - j| ≤ N_w`, built on the first forward pass and cached
   (`attention.py:339-346`). The hidden nodes are HEALPix in **ring** order
   (`nest_ordering: false`), so consecutive indices run along a latitude ring:
   the window is a zonal strip, ~11° of latitude by every longitude, not a
   compact patch. Over 4 layers that caps meridional transport at ~22°. Measured
   on the graph by `extract_dims.py`, not assumed.
2. **`lin_q` / `lin_k` / `lin_v` are plain `nn.Linear`**, not `layer_kernels`
   (`attention.py:145-147` binds the factory and then uses it only for
   `projection`), and they have `bias=False` where `projection` has bias.
3. **The residual lands after the output projection here**, unlike the mappers,
   which add `r` *before* theirs. Two conventions in one model.
4. **`num_chunks` means something different from the mappers'.** Here it splits
   the layer stack into gradient-checkpointing groups of `num_layers/num_chunks`
   (`processor.py:90`); in the mappers it partitions destination nodes. With 4
   layers and 4 chunks that is one layer per checkpoint — maximum recomputation.

## Verified decoder forward pass

Same checkout and case. `x_src` is the processor output with the latent skip
already added; `x_dst` is the **raw** `ψ` the encoder handed back.

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

1. **No source embedding at all.** `pre_process` embeds only the destination
   (`mapper.py:698-701`). Harmless while `num_channels` is the width on both
   sides, but the latent stream gets no per-mapper reprojection on the way out.
2. **`node_data_extractor` is built from `nn.LayerNorm` / `nn.Linear` directly**,
   not from `layer_kernels` — so a change to `model.layer_kernels` silently
   misses the one layer that produces the model's output.
3. **The residual add is dead here.** `_internal_output_idx` is
   `model.output.prognostic`, and with `sla` under `diagnostic:` and nothing
   prognostic, both index tensors are empty (verified on the checkpoint).
   `SkipConnection` still runs — two view ops — and its result is added to no
   column. Cheap, but the config reads as though the model has a persistence path
   and it does not. Note `prognostic` is a *fallback* category
   (`data_indices/collection.py:57-58`: everything not declared forcing,
   diagnostic or target), so adding a variable to the zarr and forgetting to
   declare it silently switches this path on.

### Where the diagnostic "comes in"

Nowhere a diagram could show. There is no separate diagnostic branch: the
distinction lives entirely in the **index sets**. `model.output` is
`prognostic ∪ diagnostic`, the extractor's Linear emits all of them at once in one
matrix, and the only thing separating the two is which columns
`_assemble_output` adds the input state back onto.

So the decoder page does not try to draw it — it shows the extractor Linear
(accented) feeding a residual `⊕` whose other input is grey, dashed and labelled
`added on P only`, and the explanation lives in the Notes section on page 4 as
prose. An explanatory box floating over the diagram was tried first and removed:
it implied there was a mechanism there to point at.

## Multi-dataset findings

For the atmosphere/forcings plan. Verified in `models/encoder_processor_decoder.py`:

- **Every dataset gets an encoder AND a decoder, unconditionally** (`:41`, `:79`,
  `:300`). No encoder-only dataset in this code path; each needs *both* edge sets,
  `(ds,"to","hidden")` and `("hidden","to",ds)`, or construction KeyErrors.
- Latents are **summed**, not concatenated or gated (`:273`). No learned weighting
  between datasets.
- `x_hidden_latent` is built **once** (`:229`) and handed to every encoder, so with
  K datasets a latent node receives K attribute-derived residual terms.
- `model.encoder` is one config instantiated per dataset (`:51`) — separate
  weights, shared hyperparameters. Whether per-dataset overrides exist is
  unverified.

| what | where | why |
|---|---|---|
| sin/cos julian day, insolation | extra channels in `ocean` under `forcing:` | same node set; costs one wider Linear |
| sin/cos lat-lon | **already in `a_x`** — don't add | `graph.py:95-98` stores `cat([sin,cos])` |
| winds / land-defined variables | own dataset | genuinely different node set |

A separate dataset on the *same* grid would duplicate all 107950 edges and add a
second attention block, and it changes the semantics: a forcing in its own dataset
arrives spatially-attended over its own neighbourhood, decoupled from the SLA at
the same location, whereas concatenated into `ψ` it travels through the same
attention weight as its node's own SLA.

## Open threads

- **Node ordering.** The hidden nodes are HEALPix ring-ordered, which is what
  makes the processor's window zonal. This is intentional.
- **Adding a figure.** Reuse the `blk` / `wire` / `shp` styles and extend `ROWS`
  in `extract_dims.py` for any new dimensions. A row lands in the notation
  table's first column unless placed after the `SPLIT` marker.
- **Other variables / datasets** — see the multi-dataset section above.
- **Land masking** — parked. The 2967 empty-neighbourhood latent nodes and
  `sea_mask` interact with loss masking; revisit if a land-inclusive dataset is
  added.

Page 5 of the PDF carries the full list of architectural oddities. The ones worth
chasing: attention aggregation is degree-normalised where GraphCast's sums, so the
output carries no magnitude cue for observation density; the decoder's receptive
field is a fixed three latent nodes regardless of local density; the processor's
is a zonal band; and nothing is pinned to the graph in v4, so turning
`trainable_size` back on would add order `N_x + N_z` and `N_EE + N_DE`
non-transferable parameters.
