PYTHON = python
MARKDOWN = markdown2
PDFLATEX = pdflatex
BIBTEX = bibtex
MKDOCS = mkdocs
SVG2PDF = svg2pdf

site=doc/site/docs

mfiles=\
vl_argparse.m \
vl_compilenn.m \
vl_nnbnorm.m \
vl_nnconcat.m \
vl_nnconv.m \
vl_nnconvt.m \
vl_nndropout.m \
vl_nnloss.m \
vl_nnnoffset.m \
vl_nnnormalize.m \
vl_nnpdist.m \
vl_nnpool.m \
vl_nnrelu.m \
vl_nnsigmoid.m \
vl_nnsoftmax.m \
vl_nnsoftmaxloss.m \
vl_nnspnorm.m \
vl_rootnn.m \
vl_setupnn.m \
simplenn/vl_simplenn.m \
simplenn/vl_simplenn_diagnose.m \
simplenn/vl_simplenn_display.m \
simplenn/vl_simplenn_move.m \
+dagnn/@DagNN/DagNN.m

mfile_src=$(addprefix matlab/src/, $(mfiles))
mfile_doc=$(patsubst %.m,%.md,$(addprefix $(site)/mfiles/, $(mfiles)))
mds=$(wildcard doc/site/docs/*.md)
svg_src=$(wildcard doc/figures/svg/*.svg)
svg_tgt=$(patsubst %.svg, %.pdf, $(svg_src))

doc: doc-site doc-manual
doc-site: doc/site/site/index.html
doc-manual: doc/matconvnet-manual.pdf

doc/matconvnet-manual.pdf : doc/matconvnet-manual.tex $(svg_tgt)
	mkdir -p doc/.build
	ln -sf ../references.bib doc/.build/references.bib
	cd "$(dir $(<))" ;\
	$(PDFLATEX) -file-line-error -output-directory=../doc/.build/ "$(notdir $(<))" ;
	cd doc/.build ; $(BIBTEX) matconvnet-manual || true ;
	cd "$(dir $(<))" ;\
	$(PDFLATEX) -file-line-error -output-directory=../doc/.build/ "$(notdir $(<))" ;\
	$(PDFLATEX) -file-line-error -output-directory=../doc/.build/ "$(notdir $(<))" ;
	cp -f doc/.build/matconvnet-manual.pdf doc/

doc/figures/svg/%.pdf : doc/figures/svg/%.svg
	$(SVG2PDF) "$(<)" "$(@)"

$(site)/mfiles/%.md : matlab/%.m $(site)/mfiles/.stamp doc/matdoc.py doc/matdocparser.py
	mkdir -p $(dir $(@))
	$(PYTHON) doc/matdoc.py "$(<)" > "$(@)"

doc/site/site/index.html : doc/site/mkdocs.yml $(mfile_doc) $(mds)
	cd doc/site ; $(MKDOCS) build --clean

doc-clean:
	rm -f doc/matdocparser.pyc doc/matdoc.pyc
	rm -f $(svg_tgt)

doc-distclean:
	rm -f doc/matconvnet-manual.pdf
	rm -rf doc/site/site
	rm -f $(mfile_doc)

doc-info:
	@echo "mds=$(mds)"
	@echo "mfile_src=$(mfile_src)"
	@echo "mfile_doc=$(mfile_doc)"
	@echo "svg_tgt=$(svg_tgt)"
