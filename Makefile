# file: Makefile
# author: Andrea Vedaldi
# brief: matconvnet makefile for mex files

# Copyright (C) 2007-14 Andrea Vedaldi and Brian Fulkerson.
# All rights reserved.
#
# This file is part of the VLFeat library and is made available under
# the terms of the BSD license (see the COPYING file).

# Code
ENABLE_GPU ?=
ENABLE_IMREADJPEG ?=
DEBUG ?=
ARCH ?= maci64
MATLABROOT ?= /Applications/MATLAB_R2014a.app
CUDAROOT ?= /Developer/NVIDIA/CUDA-5.5

# Remark: each MATLAB version requires a particular CUDA Toolkit version.
# Note that multiple CUDA Toolkits can be installed.
#MATLABROOT ?= /Applications/MATLAB_R2013b.app
#CUDAROOT ?= /Developer/NVIDIA/CUDA-5.5
#MATLABROOT ?= /Applications/MATLAB_R2014b.app
#CUDAROOT ?= /Developer/NVIDIA/CUDA-6.0

# Maintenance
NAME = matconvnet
VER = 1.0-beta8
DIST = $(NAME)-$(VER)
RSYNC = rsync
HOST = vlfeat-admin:sites/sandbox-matconvnet
GIT = git

# --------------------------------------------------------------------
#                                                        Configuration
# --------------------------------------------------------------------

# General options
MEX = $(MATLABROOT)/bin/mex
MEXEXT = $(MATLABROOT)/bin/mexext
MEXARCH = $(subst mex,,$(shell $(MEXEXT)))
MEXOPTS ?= matlab/src/config/mex_CUDA_$(ARCH).xml
MEXFLAGS = -largeArrayDims -lmwblas
MEXFLAGS_GPU = \
-DENABLE_GPU \
-f "$(MEXOPTS)" \
 $(MEXFLAGS)
SHELL = /bin/bash # sh not good enough
NVCC = $(CUDAROOT)/bin/nvcc

ifneq ($(DEBUG),)
MEXFLAGS += -g
MEXFLAGS_GPU += -g
endif

# Mac OS X Intel
ifeq "$(ARCH)" "$(filter $(ARCH),maci64)"
MEXFLAGS_GPU += -L$(CUDAROOT)/lib -lcublas -lcudart
endif

# Linux
ifeq "$(ARCH)" "$(filter $(ARCH),glnxa64)"
MEXFLAGS_GPU += -L$(CUDAROOT)/lib64 -lcublas -lcudart
endif

# --------------------------------------------------------------------
#                                                      Build MEX files
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
cpp_tgt:=$(subst matlab/src/bits/,matlab/mex/.build/,$(cpp_tgt))

.PHONY: all, distclean, clean, info, pack, post, post-doc, doc

all: $(mex_tgt)

# Create build directory
%/.stamp:
	mkdir -p $(@)/ ; touch $(@)/.stamp
$(mex_tgt): matlab/mex/.build/.stamp
$(cpp_tgt): matlab/mex/.build/.stamp

# Standard code
.PRECIOUS: matlab/mex/.build/%.o

matlab/mex/.build/%.o : matlab/src/bits/%.cpp
	$(MEX) -c $(MEXFLAGS) "$(<)"
	mv -f "$(notdir $(@))" "$(@)"

matlab/mex/.build/%.o : matlab/src/bits/%.cu
	MW_NVCC_PATH='$(NVCC)' \
	$(MEX) -c $(MEXFLAGS_GPU) "$(<)" $(nvcc_filter)
	mv -f "$(notdir $(@))" "$(@)"

# MEX code
ifneq ($(ENABLE_GPU),)
# prefer .cu over .cpp and .c when GPU is enabled; this rule must come before the following ones
matlab/mex/%.mex$(MEXARCH) : matlab/src/%.cu matlab/mex/.build/.stamp $(cpp_tgt)
	MW_NVCC_PATH='$(NVCC)' \
	$(MEX) $(MEXFLAGS_GPU) "$(<)" -output "$(@)" $(cpp_tgt) $(nvcc_filter)
endif

matlab/mex/%.mex$(MEXARCH) : matlab/src/%.c $(cpp_tgt)
	$(MEX) $(MEXFLAGS) "$(<)" -output "$(@)" $(cpp_tgt)

matlab/mex/%.mex$(MEXARCH) : matlab/src/%.cpp $(cpp_tgt)
	$(MEX) $(MEXFLAGS) "$(<)" -output "$(@)" $(cpp_tgt)

# This MEX file does not require GPU code, but requires libjpeg
matlab/mex/vl_imreadjpeg.mex$(MEXARCH): MEXFLAGS+=-I/opt/local/include -L/opt/local/lib -ljpeg
matlab/mex/vl_imreadjpeg.mex$(MEXARCH): matlab/src/vl_imreadjpeg.c
	$(MEX) $(MEXFLAGS) "$(<)" -output "$(@)"

# --------------------------------------------------------------------
#                                                        Documentation
# --------------------------------------------------------------------

include doc/Makefile

# --------------------------------------------------------------------
#                                                          Maintenance
# --------------------------------------------------------------------

info: doc-info
	@echo "mex_src=$(mex_src)"
	@echo "mex_tgt=$(mex_tgt)"
	@echo "cpp_src=$(cpp_src)"
	@echo "cpp_tgt=$(cpp_tgt)"

clean: doc-clean
	find . -name '*~' -delete
	rm -f $(cpp_tgt)
	rm -rf matlab/mex/.build

distclean: clean doc-distclean
	rm -rf matlab/mex
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
	$(RSYNC) -aP README.md doc/matconvnet-manual.pdf $(HOST)/
	$(RSYNC) -aP README.md doc/site/site/ $(HOST)/
