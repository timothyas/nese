# Encoder and decoder figures.  `make` regenerates the checkpoint-derived numbers if the
# checkpoint is newer, then builds the four-page PDF.

CKPT ?= ../../runs/v4/checkpoint/23c396bfc6ff4049a776ee6aca7af43f/inference-last.ckpt
CONDA_ENV ?= nese-anemoi
PYTHON = conda run --no-capture-output -n $(CONDA_ENV) python

GENERATED = dims.tex symbols_table.tex
SOURCES = network_diagram.tex encoder_figure.tex decoder_figure.tex notation.tex $(GENERATED)

.PHONY: all dims figure png clean

all: network_diagram.pdf

dims: $(GENERATED)

$(GENERATED): extract_dims.py $(CKPT)
	$(PYTHON) extract_dims.py $(CKPT)

network_diagram.pdf: $(SOURCES)
	latexmk -pdf -interaction=nonstopmode network_diagram.tex

# Each figure on its own, tight-cropped, for \includegraphics into a paper.
figure: encoder_standalone.pdf decoder_standalone.pdf

encoder_standalone.pdf: encoder_standalone.tex encoder_figure.tex notation.tex
	latexmk -pdf -interaction=nonstopmode encoder_standalone.tex

decoder_standalone.pdf: decoder_standalone.tex decoder_figure.tex notation.tex
	latexmk -pdf -interaction=nonstopmode decoder_standalone.tex

# gs rather than pdftoppm -- poppler is not assumed to be installed.
png: network_diagram.pdf
	gs -dNOPAUSE -dBATCH -sDEVICE=png16m -r200 \
	  -sOutputFile=network_diagram-%d.png network_diagram.pdf

clean:
	latexmk -C
	rm -f network_diagram-*.png encoder_standalone.pdf decoder_standalone.pdf
