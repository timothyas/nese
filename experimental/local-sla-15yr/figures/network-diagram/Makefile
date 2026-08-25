# Encoder figure.  `make` regenerates the checkpoint-derived numbers if the
# checkpoint is newer, then builds the two-page PDF.

CKPT ?= ../../runs/sea-mask-in-model/checkpoint/7a74d146c6cf4206a51f2217fad5bdeb/inference-last.ckpt
CONDA_ENV ?= nese-anemoi
PYTHON = conda run --no-capture-output -n $(CONDA_ENV) python

GENERATED = dims.tex symbols_table.tex
SOURCES = network_diagram.tex encoder_figure.tex notation.tex $(GENERATED)

.PHONY: all dims figure png clean

all: network_diagram.pdf

dims: $(GENERATED)

$(GENERATED): extract_dims.py $(CKPT)
	$(PYTHON) extract_dims.py $(CKPT)

network_diagram.pdf: $(SOURCES)
	latexmk -pdf -interaction=nonstopmode network_diagram.tex

# The figure on its own, tight-cropped, for \includegraphics into a paper.
figure: encoder_standalone.pdf

encoder_standalone.pdf: encoder_standalone.tex encoder_figure.tex notation.tex
	latexmk -pdf -interaction=nonstopmode encoder_standalone.tex

# gs rather than pdftoppm -- poppler is not assumed to be installed.
png: network_diagram.pdf
	gs -dNOPAUSE -dBATCH -sDEVICE=png16m -r200 \
	  -sOutputFile=network_diagram-%d.png network_diagram.pdf

clean:
	latexmk -C
	rm -f network_diagram-*.png encoder_standalone.pdf
