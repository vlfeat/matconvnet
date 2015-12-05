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
vl_imreadjpeg.m \
vl_nnbnorm.m \
vl_nnconcat.m \
vl_nnconv.m \
vl_nnconvt.m \
vl_nncrop.m \
vl_nndropout.m \
vl_nnloss.m \
vl_nnnoffset.m \
vl_nnnormalize.m \
vl_nnnormalizelp.m \
vl_nnpdist.m \
vl_nnpool.m \
vl_nnrelu.m \
vl_nnsigmoid.m \
vl_nnsoftmax.m \
vl_nnsoftmaxloss.m \
vl_nnspnorm.m \
vl_rootnn.m \
vl_setupnn.m \
vl_imreadjpeg.m \
simplenn/vl_simplenn.m \
simplenn/vl_simplenn_diagnose.m \
simplenn/vl_simplenn_tidy.m \
simplenn/vl_simplenn_display.m \
simplenn/vl_simplenn_move.m \
+dagnn/@DagNN/DagNN.m

mfile_src=$(addprefix matlab/src/, $(mfiles))
mfile_doc=$(patsubst %.m,%.md,$(addprefix $(site)/mfiles/, $(mfiles)))
mds=$(wildcard doc/site/docs/*.md)
svg_src=$(wildcard doc/figures/svg/*.svg)
svg_tgt=$(patsubst %.svg, %.pdf, $(svg_src))

# Web components
jquery_url=http://code.jquery.com/jquery-1.11.3.min.js
jquery=doc/site/lib/jquery/js/jquery-1.11.3.min.js
fontawesome_url=https://fortawesome.github.io/Font-Awesome/assets/font-awesome-4.5.0.zip
fontawesome=doc/site/lib/fontawesome/css/font-awesome.min.css
bootstrap_url=https://github.com/twbs/bootstrap/releases/download/v3.3.5/bootstrap-3.3.5-dist.zip
bootstrap=doc/site/lib/bootstrap/css/bootstrap.min.css
highlight_url=http://www.robots.ox.ac.uk/~karel/hljs
highlight=doc/site/lib/highlight/js/highlight.min.js

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

$(bootstrap):
	@echo "**************************"
	@echo "Downloading Bootstrap"
	@echo "Download URL \"$(bootstrap_url)\""
	@echo "**************************"
	mkdir -p doc/site/lib/bootstrap
	wget $(bootstrap_url) -O doc/site/lib/bootstrap/bootstrap.zip
	cd doc/site/lib/bootstrap ; unzip -o bootstrap.zip
	cd doc/site/lib/bootstrap ; cp -r bootstrap-*-dist/{fonts,css,js} .
	rm -rf doc/site/lib/bootstrap/bootstrap-*-dist
	rm doc/site/lib/bootstrap/bootstrap.zip

$(fontawesome):
	@echo "**************************"
	@echo "Downloading Fontawesome"
	@echo "Download URL \"$(fontawesome_url)\""
	@echo "**************************"
	mkdir -p doc/site/lib/fontawesome
	wget $(fontawesome_url) -O doc/site/lib/fontawesome/font-awesome.zip
	cd doc/site/lib/fontawesome ; unzip -o font-awesome.zip
	cd doc/site/lib/fontawesome ; cp -r font-awesome-*/{fonts,css} .
	rm -rf doc/site/lib/fontawesome/font-awesome-*
	rm doc/site/lib/fontawesome/font-awesome.zip

$(jquery):
	@echo "**************************"
	@echo "Downloading JQuery"
	@echo "Download URL \"$(jquery_url)\""
	@echo "**************************"
	mkdir -p doc/site/lib/jquery/js
	wget $(jquery_url) -O $(jquery)

$(highlight):
	@echo "**************************"
	@echo "Downloading Highlight"
	@echo "Download URL \"$(highlight_url)\""
	@echo "**************************"
	mkdir -p doc/site/lib/highlight/{js,css}
	wget $(highlight_url)/highlight.default.css -O doc/site/lib/highlight/css/highlight.min.css
	wget $(highlight_url)/highlight.matlab.js -O doc/site/lib/highlight/js/highlight.min.js

doc/site/site/index.html : doc/site/mkdocs.yml \
  $(mfile_doc) $(mds) \
  $(bootstrap) $(fontawesome) $(jquery) $(highlight)
	mkdir -p doc/site/theme/{css,js,fonts}
	cd doc/site/theme/css ; ln -sfv ../../lib/bootstrap/css/bootstrap.min.css ./
	cd doc/site/theme/css ; ln -sfv ../../lib/fontawesome/css/font-awesome.min.css ./
	cd doc/site/theme/css ; ln -sfv ../../lib/highlight/css/highlight.min.css ./
	cd doc/site/theme/fonts ; ln -sfv ../../lib/fontawesome/fonts/fontawesome-webfont.{eot,svg,ttf,woff,woff2} ./
	cd doc/site/theme/js ; ln -sfv ../../lib/bootstrap/js/{bootstrap.min.js,npm.js} ./
	cd doc/site/theme/js ; ln -sfv ../../lib/highlight/js/highlight.min.js ./
	cd doc/site/theme/js ; ln -sfv ../../lib/jquery/js/jquery-1.11.3.min.js ./
	cd doc/site ; $(MKDOCS) build --clean

doc-clean:
	rm -f doc/matdocparser.pyc doc/matdoc.pyc
	rm -f $(svg_tgt)

doc-distclean:
	rm -f doc/matconvnet-manual.pdf
	rm -rf doc/site/site
	rm -rf doc/site/lib
	rm -f $(mfile_doc)

doc-info:
	@echo "mds=$(mds)"
	@echo "mfile_src=$(mfile_src)"
	@echo "mfile_doc=$(mfile_doc)"
	@echo "svg_tgt=$(svg_tgt)"

doc-serve:
	cd doc/site ; $(MKDOCS) serve
