TEX := main

all: $(TEX).pdf

$(TEX).pdf: $(TEX).tex
	latexmk -pdf -interaction=nonstopmode -halt-on-error $(TEX).tex

clean:
	latexmk -C

.PHONY: all clean
