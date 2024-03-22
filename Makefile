LATEX = pdflatex
LATEX_FLAGS =

BIBTEX = bibtex
BIBTEX_FLAGS =

MAIN = main

.PHONY: all clean

all: synthnav.pdf

synthnav.pdf: Paper/$(MAIN).tex
	cd Paper; $(LATEX) $(LATEX_FLAGS) $(MAIN); \
	$(BIBTEX) $(BIBTEX_FLAGS) $(MAIN); \
	$(LATEX) $(LATEX_FLAGS) $(MAIN); \
	$(LATEX) $(LATEX_FLAGS) $(MAIN);
	

clean:
	rm -f Paper/*.aux Paper/*.log Paper/*.pdf Paper/*.bbl Paper/*.blg q.log
