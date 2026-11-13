# network_diagram.pdf from the figure sources and the generated dims.tex / symbols_table.tex.
# Regenerate those on Olivia with `sbatch --export=NONE slurm/extract_dims.sh` from the
# experiment directory.

SOURCES = network_diagram.tex encoder_figure.tex processor_figure.tex decoder_figure.tex \
          notation.tex dims.tex symbols_table.tex

.PHONY: all png clean

all: network_diagram.pdf

network_diagram.pdf: $(SOURCES)
	latexmk -pdf -interaction=nonstopmode network_diagram.tex

png: network_diagram.pdf
	gs -dNOPAUSE -dBATCH -sDEVICE=png16m -r200 \
	  -sOutputFile=network_diagram-%d.png network_diagram.pdf

clean:
	latexmk -C
	rm -f network_diagram-*.png
