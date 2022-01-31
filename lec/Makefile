TEXINPUTS:=.

LEC= \
	pdf/background.pdf \
	pdf/2022-01-24.pdf \
	pdf/2022-01-26.pdf \
	pdf/2022-01-28.pdf \
	pdf/2022-01-31.pdf \
	pdf/2022-02-02.pdf \
	pluto/2022-01-24-perf.jl \
	pluto/2022-01-24-perf.jl.html \
	pluto/2022-01-26-matrices.jl \
	pluto/2022-01-26-matrices.jl.html

.PHONY: all copy
all: $(LEC)

pdf/%.pdf: notes/%.tex
	xelatex -output-dir pdf $<
	xelatex -output-dir pdf $<

pdf/2022-01-24.pdf: notes/2022-01-24.tex
	xelatex -output-dir pdf $<
	bibtex pdf/2022-01-24
	xelatex -output-dir pdf $<
	xelatex -output-dir pdf $<

clean:
	rm -f pdf/*.aux pdf/*.log pdf/*.out
	rm -f pdf/*.nav pdf/*.snm pdf/*.toc 
	rm -f pdf/*.vrb

distclean: clean
	rm -f pdf/*.pdf

copy:
	mkdir -p ../web/lec
	cp $(LEC) ../web/lec
