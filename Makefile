# file: Makefile
# author: Andrea Vedaldi
# brief: matconvnet makefile for mex files

# Copyright (C) 2014-15 Andrea Vedaldi
# All rights reserved.
#
# This file is part of the VLFeat library and is made available under
# the terms of the BSD license (see the COPYING file).

# ENABLE_GPU -- Set to YES to enable GPU support (requires CUDA and the MATLAB Parallel Toolbox)
# ENABLE_CUDNN -- Set to YES to enable CUDNN support. This will also likely require
# ENABLE_IMREADJPEG -- Set to YES to compile the function VL_IMREADJPEG()

ENABLE_GPU ?=
ENABLE_CUDNN ?=
ENABLE_IMREADJPEG ?= yes
DEBUG ?=
ARCH ?= maci64

# Configure MATLAB
MATLABROOT ?= /Applications/MATLAB_R2014b.app

# Configure CUDA and CuDNN. CUDAMETHOD can be either 'nvcc' or 'mex'.
CUDAROOT ?= /Developer/NVIDIA/CUDA-5.5
CUDNNROOT ?= $(CURDIR)/local/
CUDAMETHOD ?= $(if $(ENABLE_CUDNN),nvcc,mex)

# Configure the image library (needed only if ENABLE_IMREADJPEG is true).
# IMAGELIB can be either 'libjpeg' (default on Linux) or 'quartz' (default on a Mac)
IMAGELIB ?= $(IMAGELIB_DEFAULT)
IMAGELIB_CFLAGS ?= $(IMAGELIB_CFLAGS_DEFAULT)
IMAGELIB_LDFLAGS ?= $(IMAGELIB_LDFLAGS_DEFAULT)

# Remark: each MATLAB version requires a particular CUDA Toolkit version.
# Note that multiple CUDA Toolkits can be installed.
#MATLABROOT ?= /Applications/MATLAB_R2014b.app
#CUDAROOT ?= /Developer/NVIDIA/CUDA-6.0
#MATLABROOT ?= /Applications/MATLAB_R2015b.app
#CUDAROOT ?= /Developer/NVIDIA/CUDA-6.5

# Maintenance
NAME = matconvnet
VER = 1.0-beta16
DIST = $(NAME)-$(VER)
LATEST = $(NAME)-latest
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
MEXFLAGS = -cxx -largeArrayDims -lmwblas \
$(if $(ENABLE_GPU),-DENABLE_GPU,) \
$(if $(ENABLE_CUDNN),-DENABLE_CUDNN -I$(CUDNNROOT),)
MEXFLAGS_GPU = $(MEXFLAGS) -f "$(MEXOPTS)"
SHELL = /bin/bash # sh not good enough

NVCC = $(CUDAROOT)/bin/nvcc
NVCCVER = $(shell $(NVCC) --version | \
sed -n 's/.*V\([0-9]*\).\([0-9]*\).\([0-9]*\).*/\1 \2 \3/p' | \
xargs printf '%02d%02d%02d')
NVCCVER_LT_70 = $(shell test $(NVCCVER) -lt 070000 && echo true)

# this is used *onyl* for the 'nvcc' method
NVCCFLAGS = \
-gencode=arch=compute_30,code=\"sm_30,compute_30\" \
-DENABLE_GPU \
$(if $(ENABLE_CUDNN),-DENABLE_CUDNN -I$(CUDNNROOT),) \
-I"$(MATLABROOT)/extern/include" \
-I"$(MATLABROOT)/toolbox/distcomp/gpu/extern/include" \
-Xcompiler -fPIC
MEXFLAGS_NVCC = $(MEXFLAGS) -lmwgpu

ifneq ($(DEBUG),)
MEXFLAGS += -g
NVCCFLAGS += -g -O0
else
MEXFLAGS += -DNDEBUG -O
NVCCFLAGS += -DNDEBUG -O3
# we still want debug symbols
MEXFLAGS += CXXOPTIMFLAGS='$$CXXOPTIMFLAGS -g'
MEXFLAGS += LDOPTIMFLAGS='$$LDOPTIMFLAGS -g'
NVCCFLAGS += -g
endif

ifdef VERB
MEXFLAGS += -v
NVCCFLAGS += -v
endif

# Mac OS X Intel
ifeq "$(ARCH)" "$(filter $(ARCH),maci64)"
MEXFLAGS_GPU += -L$(CUDAROOT)/lib
ifeq ($(NVCCVER_LT_70),true)
# if using an old version of CUDA
MEXFLAGS_NVCC += -L$(CUDAROOT)/lib LDFLAGS='$$LDFLAGS -stdlib=libstdc++'
else
MEXFLAGS_NVCC += -L$(CUDAROOT)/lib LDFLAGS='$$LDFLAGS'
endif
IMAGELIB_DEFAULT = quartz
endif

# Linux
ifeq "$(ARCH)" "$(filter $(ARCH),glnxa64)"
MEXFLAGS_GPU += -L$(CUDAROOT)/lib64
MEXFLAGS_NVCC += -L$(CUDAROOT)/lib64
IMAGELIB_DEFAULT = libjpeg
MEXFLAGS += CXXOPTIMFLAGS='$$CXXOPTIMFLAGS -mssse3 -ftree-vect-loop-version -ffast-math -funroll-all-loops'
NVCCFLAGS += -Xcompiler -mssse3,-ftree-vect-loop-version,-ffast-math,-funroll-all-loops
endif

# Image library
ifeq ($(IMAGELIB),libjpeg)
IMAGELIB_CFLAGS_DEFAULT :=
IMAGELIB_LDFLAGS_DEFAULT := -ljpeg
endif
ifeq ($(IMAGELIB),quartz)
IMAGELIB_CFLAGS_DEFAULT :=
IMAGELIB_LDFLAGS_DEFAULT := LDFLAGS='$$LDFLAGS -framework Cocoa -framework ImageIO'
endif
ifdef ENABLE_IMREADJPEG
MEXFLAGS += $(IMAGELIB_CFLAGS) $(IMAGELIB_LDFLAGS)
endif

MEXFLAGS_GPU += -lcublas -lcudart $(if $(ENABLE_CUDNN),-L$(CUDNNROOT) -lcudnn,)
MEXFLAGS_NVCC += -lcublas -lcudart $(if $(ENABLE_CUDNN),-L$(CUDNNROOT) -lcudnn,)

# --------------------------------------------------------------------
#                                                      Build MEX files
# --------------------------------------------------------------------

nvcc_filter=2> >( sed 's/^\(.*\)(\([0-9][0-9]*\)): \([ew].*\)/\1:\2: \3/g' >&2 )
cpp_src :=
mex_src :=

# Files that are compiled as CPP or GPU (CUDA) depending on whether GPU support
# is enabled.
ext := $(if $(ENABLE_GPU),cu,cpp)
cpp_src+=matlab/src/bits/data.$(ext)
cpp_src+=matlab/src/bits/datamex.$(ext)
cpp_src+=matlab/src/bits/nnconv.$(ext)
cpp_src+=matlab/src/bits/nnbias.$(ext)
cpp_src+=matlab/src/bits/nnfullyconnected.$(ext)
cpp_src+=matlab/src/bits/nnsubsample.$(ext)
cpp_src+=matlab/src/bits/nnpooling.$(ext)
cpp_src+=matlab/src/bits/nnnormalize.$(ext)
cpp_src+=matlab/src/bits/nnbnorm.$(ext)
mex_src+=matlab/src/vl_nnconv.$(ext)
mex_src+=matlab/src/vl_nnconvt.$(ext)
mex_src+=matlab/src/vl_nnpool.$(ext)
mex_src+=matlab/src/vl_nnnormalize.$(ext)
mex_src+=matlab/src/vl_nnbnorm.$(ext)
ifdef ENABLE_IMREADJPEG
mex_src+=matlab/src/vl_imreadjpeg.cpp
endif

# CPU-specific files
cpp_src+=matlab/src/bits/impl/im2row_cpu.cpp
cpp_src+=matlab/src/bits/impl/subsample_cpu.cpp
cpp_src+=matlab/src/bits/impl/copy_cpu.cpp
cpp_src+=matlab/src/bits/impl/pooling_cpu.cpp
cpp_src+=matlab/src/bits/impl/normalize_cpu.cpp
cpp_src+=matlab/src/bits/impl/bnorm_cpu.cpp
cpp_src+=matlab/src/bits/impl/tinythread.cpp
ifdef ENABLE_IMREADJPEG
cpp_src+=matlab/src/bits/impl/imread_$(IMAGELIB).cpp
endif

# GPU-specific files
ifdef ENABLE_GPU
cpp_src+=matlab/src/bits/impl/im2row_gpu.cu
cpp_src+=matlab/src/bits/impl/subsample_gpu.cu
cpp_src+=matlab/src/bits/impl/copy_gpu.cu
cpp_src+=matlab/src/bits/impl/pooling_gpu.cu
cpp_src+=matlab/src/bits/impl/normalize_gpu.cu
cpp_src+=matlab/src/bits/impl/bnorm_gpu.cu
cpp_src+=matlab/src/bits/datacu.cu
ifdef ENABLE_CUDNN
cpp_src+=matlab/src/bits/impl/nnconv_cudnn.cu
cpp_src+=matlab/src/bits/impl/nnpooling_cudnn.cu
cpp_src+=matlab/src/bits/impl/nnbias_cudnn.cu
endif
endif

mex_tgt:=$(subst matlab/src/,matlab/mex/,$(mex_src))
mex_tgt:=$(patsubst %.cpp,%.mex$(MEXARCH),$(mex_tgt))
mex_tgt:=$(patsubst %.cu,%.mex$(MEXARCH),$(mex_tgt))

cpp_tgt:=$(patsubst %.cpp,%.o,$(cpp_src))
cpp_tgt:=$(patsubst %.cu,%.o,$(cpp_tgt))
cpp_tgt:=$(subst matlab/src/bits/,matlab/mex/.build/,$(cpp_tgt))

.PHONY: all, distclean, clean, info, pack, post, post-doc, doc

all: $(cpp_tgt) $(mex_tgt)

# Create build directory
%/.stamp:
	mkdir -p $(*)/ ; touch $(*)/.stamp
$(mex_tgt): matlab/mex/.build/impl/.stamp
$(cpp_tgt): matlab/mex/.build/impl/.stamp
$(cu_tgt): matlab/mex/.build/impl/.stamp

# Standard code
.PRECIOUS: matlab/mex/.build/%.o
.PRECIOUS: %/.stamp

ifeq ($(CUDAMETHOD),mex)
include Makefile.mex
else
include Makefile.nvcc
endif

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
	@echo "cu_src=$(cu_src)"
	@echo "cu_tgt=$(cu_tgt)"
	@echo '------------------------------'
	@echo 'MEXFLAGS=$(MEXFLAGS)'
	@echo 'MEXFLAGS_GPU=$(MEXFLAGS_GPU)'
	@echo 'MEXFLAGS_NVCC=$(MEXFLAGS_NVCC)'
	@echo '------------------------------'
	@echo 'NVCC=$(NVCC)'
	@echo 'NVCCVER=$(NVCCVER)'
	@echo 'NVCCVER_LT_70=$(NVCCVER_LT_70)'
	@echo 'NVCCFLAGS=$(NVCCFLAGS)'


clean: doc-clean
	find . -name '*~' -delete
	rm -f $(cpp_tgt) $(cu_tgt)
	rm -rf matlab/mex/.build

distclean: clean doc-distclean
	rm -rf matlab/mex
	rm -f doc/index.html doc/matconvnet-manual.pdf
	rm -f $(NAME)-*.tar.gz

pack:
	COPYFILE_DISABLE=1 \
	COPY_EXTENDED_ATTRIBUTES_DISABLE=1 \
	$(GIT) archive --prefix=$(NAME)-$(VER)/ v$(VER) | gzip > $(DIST).tar.gz
	ln -sf $(DIST).tar.gz $(LATEST).tar.gz

post: pack
	$(RSYNC) -aP $(DIST).tar.gz $(LATEST).tar.gz $(HOST)/download/

post-models:
	$(RSYNC) -aP data/models/*.mat $(HOST)/models/

post-doc: doc
	$(RSYNC) -aP README.md doc/matconvnet-manual.pdf $(HOST)/
	$(RSYNC) -aP README.md doc/site/site/ $(HOST)/
