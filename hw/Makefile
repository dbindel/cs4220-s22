TEXINPUTS:=.

HW= \
	hw1.jl \
	hw2.jl \
	proj1.jl \
	hw1.jl.html \
	hw2.jl.html \
	proj1.jl.html \
	roadNet-CA.txt

.PHONY: all copy
all: $(LEC)

pdf/%.pdf: notes/%.tex
	xelatex -output-dir pdf $<
	xelatex -output-dir pdf $<

clean:
	rm -f pdf/*.aux pdf/*.log pdf/*.out
	rm -f pdf/*.nav pdf/*.snm pdf/*.toc 
	rm -f pdf/*.vrb

distclean: clean
	rm -f pdf/*.pdf

copy:
	mkdir -p ../web/hw
	cp $(HW) ../web/hw
