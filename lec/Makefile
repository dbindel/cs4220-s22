TEXINPUTS:=.

LEC= \
	pdf/background.pdf \
	pdf/2022-01-24.pdf \
	pdf/2022-01-26.pdf \
	pdf/2022-01-28.pdf \
	pdf/2022-01-31.pdf \
	pdf/2022-02-02.pdf \
	pdf/2022-02-09.pdf \
	pdf/2022-02-11.pdf \
	pdf/2022-02-14.pdf \
	pdf/2022-02-16.pdf \
	pdf/2022-02-21.pdf \
	pdf/2022-02-23.pdf \
	pdf/2022-02-25.pdf \
	pdf/2022-03-04.pdf \
	pdf/2022-03-07.pdf \
	pdf/2022-03-09.pdf \
	pdf/2022-03-14.pdf \
	pdf/2022-03-16.pdf \
	pdf/2022-03-18.pdf \
	pdf/2022-03-21.pdf \
	pdf/2022-03-23.pdf \
	pdf/2022-03-25.pdf \
	pluto/2022-01-24-perf.jl \
	pluto/2022-01-24-perf.jl.html \
	pluto/2022-01-26-matrices.jl \
	pluto/2022-01-26-matrices.jl.html \
	pluto/2022-02-02.jl \
	pluto/2022-02-02.jl.html \
	pluto/2022-02-09.jl \
	pluto/2022-02-09.jl.html \
	pluto/2022-02-11.jl \
	pluto/2022-02-11.jl.html \
	pluto/2022-02-14.jl \
	pluto/2022-02-14.jl.html \
	pluto/2022-02-18.jl \
	pluto/2022-02-18.jl.html \
	pluto/2022-02-21.jl \
	pluto/2022-02-21.jl.html \
	pluto/2022-02-23.jl \
	pluto/2022-02-23.jl.html \
	pluto/2022-02-25.jl \
	pluto/2022-02-25.jl.html \
	pluto/2022-03-07.jl \
	pluto/2022-03-07.jl.html \
	pluto/2022-03-09.jl \
	pluto/2022-03-09.jl.html \
	pluto/2022-03-14.jl \
	pluto/2022-03-14.jl.html \
	pluto/2022-03-16.jl \
	pluto/2022-03-16.jl.html \
	pluto/2022-03-21.jl \
	pluto/2022-03-21.jl.html \
	pluto/2022-03-23.jl \
	pluto/2022-03-23.jl.html \
	pluto/2022-03-25.jl \
	pluto/2022-03-25.jl.html

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
