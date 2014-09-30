# file: Makefile
# author: Andrea Vedaldi
# brief: matconvnet makefile for mex files

# Copyright (C) 2007-14 Andrea Vedaldi and Brian Fulkerson.
# All rights reserved.
#
# This file is part of the VLFeat library and is made available under
# the terms of the BSD license (see the COPYING file).

# For Linux: intermediate files must be compiled with -fPIC to go in a MEX file

ENABLE_GPU ?=
ENABLE_IMREADJPEG ?=
DEBUG ?=
ARCH ?= maci64
CUDAROOT ?= /Developer/NVIDIA/CUDA-5.5
MATLABROOT ?= /Applications/MATLAB_R2014a.app

NAME = matconvnet
VER = 1.0-beta7
DIST = $(NAME)-$(VER)
MARKDOWN = markdown2
HOST = vlfeat-admin:sites/sandbox-matconvnet
GIT = git
PDFLATEX = pdflatex
BIBTEX = bibtex
RSYNC = rsync

# --------------------------------------------------------------------
#                                                        Configuration
# --------------------------------------------------------------------

# General options
MEX = $(MATLABROOT)/bin/mex
MEXEXT = $(MATLABROOT)/bin/mexext
MEXARCH = $(subst mex,,$(shell $(MEXEXT)))
MEXOPTS = -largeArrayDims -lmwblas
MEXOPTS_GPU = \
-DENABLE_GPU -f matlab/src/config/mex_CUDA_$(ARCH).xml \
-largeArrayDims -lmwblas
SHELL = /bin/bash # sh not good enough

# at least compute 2.0 required
NVCC = $(CUDAROOT)/bin/nvcc
NVCCOPTS = \
-gencode=arch=compute_20,code=sm_21 \
-gencode=arch=compute_30,code=sm_30

ifneq ($(DEBUG),)
MEXOPTS += -g
NVCCOPTS += -g
endif

# Mac OS X Intel
ifeq "$(ARCH)" "$(filter $(ARCH),maci64)"
MEXOPTS_GPU += -L$(CUDAROOT)/lib -lcublas -lcudart
endif

# Linux
ifeq "$(ARCH)" "$(filter $(ARCH),glnxa64)"
NVCCOPTS += --compiler-options=-fPIC
MEXOPTS_GPU += -L$(CUDAROOT)/lib64 -lcublas -lcudart
endif

# --------------------------------------------------------------------
#                                                           Do the job
# --------------------------------------------------------------------

nvcc_filter=2> >( sed 's/^\(.*\)(\([0-9][0-9]*\)): \([ew].*\)/\1:\2: \3/g' >&2 )

cpp_src:=matlab/src/bits/im2col.cpp
cpp_src+=matlab/src/bits/pooling.cpp
cpp_src+=matlab/src/bits/normalize.cpp
cpp_src+=matlab/src/bits/subsample.cpp

ifneq ($(ENABLE_IMREADJPEG),)
mex_src:=matlab/src/vl_imreadjpeg.c
endif

ifeq ($(ENABLE_GPU),)
mex_src+=matlab/src/vl_nnconv.cpp
mex_src+=matlab/src/vl_nnpool.cpp
mex_src+=matlab/src/vl_nnnormalize.cpp
else
mex_src+=matlab/src/vl_nnconv.cu
mex_src+=matlab/src/vl_nnpool.cu
mex_src+=matlab/src/vl_nnnormalize.cu
cpp_src+=matlab/src/bits/im2col_gpu.cu
cpp_src+=matlab/src/bits/pooling_gpu.cu
cpp_src+=matlab/src/bits/normalize_gpu.cu
cpp_src+=matlab/src/bits/subsample_gpu.cu
endif

mex_tgt:=$(subst matlab/src/,matlab/mex/,$(mex_src))
mex_tgt:=$(patsubst %.c,%.mex$(MEXARCH),$(mex_tgt))
mex_tgt:=$(patsubst %.cpp,%.mex$(MEXARCH),$(mex_tgt))
mex_tgt:=$(patsubst %.cu,%.mex$(MEXARCH),$(mex_tgt))

cpp_tgt:=$(patsubst %.cpp,%.o,$(cpp_src))
cpp_tgt:=$(patsubst %.cu,%.o,$(cpp_tgt))

.PHONY: all, distclean, clean, info, pack, post, post-doc, doc

all: $(mex_tgt)

matlab/mex/.stamp:
	mkdir matlab/mex ; touch matlab/mex/.stamp

# Standard code
matlab/src/bits/%.o : matlab/src/bits/%.cpp
	$(MEX) -c $(MEXOPTS) "$(<)"
	mv -f "$(notdir $(@))" "$(@)"

matlab/src/bits/%.o : matlab/src/bits/%.cu
	$(NVCC) -c $(NVCCOPTS) "$(<)" -o "$(@)" $(nvcc_filter)

# MEX files
matlab/mex/vl_imreadjpeg.mex$(MEXARCH): MEXOPTS+=-I/opt/local/include -L/opt/local/lib -ljpeg

matlab/mex/%.mex$(MEXARCH) : matlab/src/%.c matlab/mex/.stamp $(cpp_tgt)
	$(MEX) $(MEXOPTS) "$(<)" -output "$(@)" $(cu_tgt) $(nvcc_filter)

matlab/mex/%.mex$(MEXARCH) : matlab/src/%.cpp matlab/mex/.stamp $(cpp_tgt)
	$(MEX) $(MEXOPTS) "$(<)" -output "$(@)" $(cu_tgt) $(nvcc_filter)

matlab/mex/%.mex$(MEXARCH) : matlab/src/%.cu matlab/mex/.stamp $(cpp_tgt)
ifeq ($(ENABLE_GPU),)
	echo "#include \"../src/$(notdir $(<))\"" > "matlab/mex/$(*).cpp"
	$(MEX) $(MEXOPTS) \
	  "matlab/mex/$(*).cpp" $(cpp_tgt) \
	  -output "$(@)" \
	  $(nvcc_filter)
	rm -f "matlab/mex/$(*).cpp"
else
	echo $(@)
	MW_NVCC_PATH='$(NVCC)' $(MEX) \
	   -output "$(@)" \
	   "$(<)" $(cpp_tgt) \
	   $(MEXOPTS_GPU) \
	   $(nvcc_filter)
endif

doc: doc/index.html doc/matconvnet-manual.pdf

doc/matconvnet-manual.pdf : doc/matconvnet-manual.tex
	mkdir -p doc/.build
	ln -sf ../references.bib doc/.build/references.bib
	$(PDFLATEX) -file-line-error -output-directory=doc/.build/ "$(<)"
	cd doc/.build ; $(BIBTEX) matconvnet-manual
	$(PDFLATEX) -file-line-error -output-directory=doc/.build/ "$(<)"
	$(PDFLATEX) -file-line-error -output-directory=doc/.build/ "$(<)"
	cp -f doc/.build/matconvnet-manual.pdf doc/

doc/index.html : doc/.build/index.html.raw doc/template.html
	sed -e '/%MARKDOWN%/{r doc/.build/index.html.raw' -e 'd;}' doc/template.html  > $(@)

doc/.build/index.html.raw : doc/index.md
	mkdir -p doc/.build
	$(MARKDOWN) -x tables $(<) > $(@)

# Other targets
info:
	@echo "mex_src=$(mex_src)"
	@echo "mex_tgt=$(mex_tgt)"
	@echo "cpp_src=$(cpp_src)"
	@echo "cpp_tgt=$(cpp_tgt)"

clean:
	find . -name '*~' -delete
	rm -f $(cpp_tgt)
	rm -rf doc/.build

distclean: clean
	rm -rf matlab/mex/
	rm -f doc/index.html doc/matconvnet-manual.pdf
	rm -f $(NAME)-*.tar.gz

pack:
	COPYFILE_DISABLE=1 \
	COPY_EXTENDED_ATTRIBUTES_DISABLE=1 \
	$(GIT) archive --prefix=$(NAME)-$(VER)/ v$(VER) | gzip > $(DIST).tar.gz

post: pack
	$(RSYNC) -aP $(DIST).tar.gz $(HOST)/download/

post-models:
	$(RSYNC) -aP data/models/*.mat $(HOST)/models/

post-doc: doc
	$(RSYNC) -aP README.md doc/{index.html,matconvnet-manual.pdf} $(HOST)/
