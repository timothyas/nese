# Network diagrams

> Picking this back up? Start with `network-illustration.md` — conventions,
> verified forward pass, open threads.

TikZ figures of the encoder and the decoder of the `AnemoiModelEncProcDec`
trained here — a `GraphTransformerForwardMapper` and a
`GraphTransformerBackwardMapper` — drawn at the block level of *Attention Is All
You Need* Fig. 1, with every wire annotated by its array shape.  The processor
is not drawn yet.

    make            # build network_diagram.pdf (figures + notation + notes)
    make figure     # encoder_standalone.pdf and decoder_standalone.pdf
    make png        # rasterise for a quick look
    make clean

    make CKPT=path/to/other/inference-last.ckpt      # a different run

`CKPT` defaults to the `v4` run, which is the one `../../training.yaml`
describes.  Earlier runs differ in ways the pages report — `v0`, for instance,
has trainable node and edge attributes and an unmasked data mesh — so the wording
on the notation and notes pages is generated from the checkpoint rather than
typed in, and follows whichever one you build against.

## Files

| file | |
|---|---|
| `encoder_figure.tex`   | the encoder `tikzpicture`. **One of the two files you edit.** |
| `decoder_figure.tex`   | the decoder `tikzpicture`. **The other one.** |
| `notation.tex`         | the symbol macros --- the one place a symbol is renamed |
| `network_diagram.tex`  | five-page landscape wrapper: encoder, decoder, notation table, notes, experiment candidates |
| `encoder_standalone.tex`, `decoder_standalone.tex` | one figure alone, for `\includegraphics` into a paper |
| `extract_dims.py`      | reads the dimensions out of a checkpoint |
| `dims.tex`             | **generated** --- `\newcommand{\valNx}{33469}` etc. |
| `symbols_table.tex`    | **generated** --- the rows of the notation table, in two columns |

The figures are drawn entirely in symbols; no number appears on pages 1--2.  The
concrete values live only in the notation table, and `extract_dims.py` reads
them off the live modules in the checkpoint, so they cannot drift from the
trained model.  It asserts that `N_psi == N_T * N_u + N_ax`, that
`N_phi == N_H * N_c`, that `N_y == N_P + N_D`, that the decoder's extractor still
emits the model's output width and that the backward mapper still has no source
embedding --- so a config or upstream change that breaks what the figures assume
fails loudly instead of drawing a lie.  It also summarises the incoming-degree
distribution of both graphs (`\valdegmin` ... `\valdegzero` for the encoder,
`\valdecdegmin` ... `\valdecdegzero` for the decoder), which the notation
caption quotes --- so those numbers track the graph rather than being typed in.
Whether the attributes carry a trainable tail is likewise read off the
checkpoint, and `\valtrainablenote` carries the wording for either case.

Both figures place every node off the `\row*` / `\col*` registers at the top of
the file, so the layout is reflowed by editing those numbers rather than by
moving nodes.  The three lanes (data, edges, latent) each carry one accent colour
from `notebooks/nice.mplstyle`, matching
`$WORK/nested-eagle-paper/figures/nested-illustration`; a fourth accent is used
only for the decoder's output extractor and its output channels.  Paths that are
present in the code but inert in this configuration are drawn grey and dashed.

The two pages share a lane convention: the **source** lane is on top and supplies
`k` and `v`, the edge lane is in the middle, the **destination** lane is at the
bottom and supplies `q` and `r`, and the destination stream leaves along the
middle line.  So the lanes keep their colours between the pages but swap
position, because the two spaces have swapped roles.

Lane labels are anchored to their own input box (`(psiin.north west)`), not to a
fixed offset off the lane row, so a box that grows a line cannot hide one.

Apparent text size is set by the aspect ratio, not the font size --- both figures
are placed with `\resizebox{\textwidth}{!}`.  The decoder's chain is longer than
the encoder's (it carries the extractor and the residual add), so its lanes are
spread further apart to compensate.  Push them too far and the page-2 caption
falls off the page.

## Extending to the processor

Add `processor_figure.tex` alongside the other two, reuse the same `blk` / `wire`
/ `shp` styles, and extend `ROWS` in `extract_dims.py` with whatever new
dimensions it needs.  A row added to `ROWS` lands in the notation table's first
column unless it is placed after the `SPLIT` marker; the table is set as two
tabulars because a single one tall enough to overflow the page is silently
shipped as an overfull box on a page of its own.
