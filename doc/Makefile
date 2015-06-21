PYTHON = python
MARKDOWN = markdown2
PDFLATEX = pdflatex
BIBTEX = bibtex
MKDOCS = mkdocs

site=doc/site/docs

mfiles=\
vl_argparse.m \
vl_compilenn.m \
vl_nnbnorm.m \
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
vl_simplenn.m \
vl_simplenn_diagnose.m \
vl_simplenn_display.m \
vl_simplenn_move.m

mfile_src=$(addprefix matlab/src/, $(mfiles))
mfile_doc=$(patsubst %.m,%.md,$(addprefix $(site)/mfiles/, $(mfiles)))
mds=$(wildcard doc/site/docs/*.md)

doc: doc-site doc-manual
doc-site: doc/site/site/index.html
doc-manual: doc/matconvnet-manual.pdf

doc/matconvnet-manual.pdf : doc/matconvnet-manual.tex
	mkdir -p doc/.build
	ln -sf ../references.bib doc/.build/references.bib
	cd "$(dir $(<))" ;\
	$(PDFLATEX) -file-line-error -output-directory=../doc/.build/ "$(notdir $(<))" ;
	cd doc/.build ; $(BIBTEX) matconvnet-manual || true ;
	cd "$(dir $(<))" ;\
	$(PDFLATEX) -file-line-error -output-directory=../doc/.build/ "$(notdir $(<))" ;\
	$(PDFLATEX) -file-line-error -output-directory=../doc/.build/ "$(notdir $(<))" ;
	cp -f doc/.build/matconvnet-manual.pdf doc/

$(site)/mfiles/%.md : matlab/%.m $(site)/mfiles/.stamp doc/matdoc.py doc/matdocparser.py
	$(PYTHON) doc/matdoc.py "$(<)" > "$(@)"

doc/site/site/index.html : doc/site/mkdocs.yml $(mfile_doc) $(mds)
	cd doc/site ; $(MKDOCS) build --clean

doc-clean:
	rm -f doc/matdocparser.pyc doc/matdoc.pyc

doc-distclean:
	rm -f doc/matconvnet-manual.pdf
	rm -rf doc/site/site

doc-info:
	@echo "mds=$(mds)"
	@echo "mfile_src=$(mfile_src)"
	@echo "mfile_doc=$(mfile_doc)"
