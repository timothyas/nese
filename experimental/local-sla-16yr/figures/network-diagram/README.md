# Encoder network diagram

> Picking this back up? Start with `network-illustration.md` — conventions,
> verified forward pass, open threads.

A TikZ figure of the encoder of the `AnemoiModelEncProcDec` trained in
`experimental/local-sla-15yr` --- a `GraphTransformerForwardMapper` --- drawn at
the block level of *Attention Is All You Need* Fig. 1, with every wire annotated
by its array shape.

    make            # build network_diagram.pdf (figure + notation + notes)
    make figure     # encoder_standalone.pdf, tight-cropped for a paper
    make png        # rasterise for a quick look
    make clean

    make CKPT=path/to/other/inference-last.ckpt      # a different run

## Files

| file | |
|---|---|
| `encoder_figure.tex`   | the `tikzpicture`. **This is the file you edit.** |
| `notation.tex`         | the symbol macros --- the one place a symbol is renamed |
| `network_diagram.tex`  | three-page landscape wrapper: figure, notation table, notes |
| `encoder_standalone.tex` | the figure alone, for `\includegraphics` into a paper |
| `extract_dims.py`      | reads the dimensions out of a checkpoint |
| `dims.tex`             | **generated** --- `\newcommand{\valNx}{49152}` etc. |
| `symbols_table.tex`    | **generated** --- the rows of the notation table |

The figure is drawn entirely in symbols; no number appears on page 1.  The
concrete values live only in the notation table, and `extract_dims.py` reads
them off the live modules in the checkpoint, so they cannot drift from the
trained model.  It asserts that `N_psi == N_T * N_u + N_ax` and that
`N_phi == N_H * N_c`, so a config change that breaks the assumed input layout
fails loudly instead of drawing a lie.  It also summarises the incoming-degree
distribution of the encoder graph (`\valdegmin`, `\valdegmax`, `\valdegmean`,
`\valdegzero`), which the notation caption quotes --- so those numbers track the
graph rather than being typed in.

`encoder_figure.tex` places every node off the `\row*` / `\col*` registers at the
top of the file, so the layout is reflowed by editing those numbers rather than
by moving nodes.  The three lanes (data, edges, latent) each carry one accent
colour from `notebooks/nice.mplstyle`, matching
`$WORK/nested-eagle-paper/figures/nested-illustration`.

## Extending to the processor and decoder

`notation.tex` already reserves `N_DE` for the decoder's edge count.  Add
`processor_figure.tex` / `decoder_figure.tex` alongside `encoder_figure.tex`,
reuse the same `blk` / `wire` / `shp` styles, and extend `ROWS` in
`extract_dims.py` with whatever new dimensions they need.
