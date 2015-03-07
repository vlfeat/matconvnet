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
# ENABLE_IMREADJPEG -- Set to YES to compile the function VL_IMREADJPEG() (requires LIBJPEG)

ENABLE_GPU ?=
ENABLE_CUDNN ?=
ENABLE_IMREADJPEG ?=
DEBUG ?=
ARCH ?= maci64
MATLABROOT ?= /Applications/MATLAB_R2014a.app
CUDAROOT ?= /Developer/NVIDIA/CUDA-5.5
CUDNNROOT ?= $(CURDIR)/local/
CUDAMETHOD ?= $(if $(ENABLE_CUDNN),nvcc,mex)
LIBJPEG_INCLUDE ?= /opt/local/include
LIBJPEG_LIB ?= /opt/local/lib

# Remark: each MATLAB version requires a particular CUDA Toolkit version.
# Note that multiple CUDA Toolkits can be installed.
#MATLABROOT ?= /Applications/MATLAB_R2013b.app
#CUDAROOT ?= /Developer/NVIDIA/CUDA-5.5
#MATLABROOT ?= /Applications/MATLAB_R2014b.app
#CUDAROOT ?= /Developer/NVIDIA/CUDA-6.0

# Maintenance
NAME = matconvnet
VER = 1.0-beta9
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
MEXFLAGS = -largeArrayDims -lmwblas \
$(if $(ENABLE_GPU),-DENABLE_GPU,) \
$(if $(ENABLE_CUDNN),-DENABLE_CUDNN -I$(CUDNNROOT),)
MEXFLAGS_GPU = $(MEXFLAGS) -f "$(MEXOPTS)"
SHELL = /bin/bash # sh not good enough
NVCC = $(CUDAROOT)/bin/nvcc

# this is used *onyl* for the 'nvcc' method
NVCCFLAGS = \
-gencode=arch=compute_30,code=\"sm_30,compute_30\" \
-DENABLE_GPU \
$(if $(ENABLE_CUDNN),-DENABLE_CUDNN -I$(CUDNNROOT),) \
-I"$(MATLABROOT)/extern/include" \
-I"$(MATLABROOT)/toolbox/distcomp/gpu/extern/include" \
-Xcompiler -fPIC
MEXFLAGS_NVCC = $(MEXFLAGS) -cxx -lmwgpu

ifneq ($(DEBUG),)
MEXFLAGS += -g
MEXFLAGS_GPU += -g
NVCCFLAGS += -g -O0
else
MEXFLAGS += -DNDEBUG -O
MEXFLAGS_GPU += -DNDEBUG -O
NVCCFLAGS += -DNDEBUG -O3
endif

# Mac OS X Intel
ifeq "$(ARCH)" "$(filter $(ARCH),maci64)"
MEXFLAGS_GPU += -L$(CUDAROOT)/lib
MEXFLAGS_NVCC += -L$(CUDAROOT)/lib LDFLAGS='$$LDFLAGS -stdlib=libstdc++'
endif

# Linux
ifeq "$(ARCH)" "$(filter $(ARCH),glnxa64)"
MEXFLAGS_GPU += -L$(CUDAROOT)/lib64
MEXFLAGS_NVCC += -L$(CUDAROOT)/lib64
endif

MEXFLAGS_GPU += -lcublas -lcudart $(if $(ENABLE_CUDNN),-L$(CUDNNROOT) -lcudnn,)
MEXFLAGS_NVCC += -lcublas -lcudart $(if $(ENABLE_CUDNN),-L$(CUDNNROOT) -lcudnn,)

# --------------------------------------------------------------------
#                                                      Build MEX files
# --------------------------------------------------------------------

nvcc_filter=2> >( sed 's/^\(.*\)(\([0-9][0-9]*\)): \([ew].*\)/\1:\2: \3/g' >&2 )
cpp_src :=
mex_src :=

# Files that are compiled as CPP or CU depending on whether GPU support
# is enabled.
ext := $(if $(ENABLE_GPU),cu,cpp)
cpp_src+=matlab/src/bits/data.$(ext)
cpp_src+=matlab/src/bits/datamex.$(ext)
cpp_src+=matlab/src/bits/nnconv.$(ext)
cpp_src+=matlab/src/bits/nnfullyconnected.$(ext)
cpp_src+=matlab/src/bits/nnsubsample.$(ext)
cpp_src+=matlab/src/bits/nnpooling.$(ext)
cpp_src+=matlab/src/bits/nnnormalize.$(ext)
mex_src+=matlab/src/vl_nnconv.$(ext)
mex_src+=matlab/src/vl_nnpool.$(ext)
mex_src+=matlab/src/vl_nnnormalize.$(ext)

# CPU-specific files
cpp_src+=matlab/src/bits/impl/im2row_cpu.cpp
cpp_src+=matlab/src/bits/impl/subsample_cpu.cpp
cpp_src+=matlab/src/bits/impl/copy_cpu.cpp
cpp_src+=matlab/src/bits/impl/pooling_cpu.cpp
cpp_src+=matlab/src/bits/impl/normalize_cpu.cpp

# GPU-specific files
ifneq ($(ENABLE_GPU),)
cpp_src+=matlab/src/bits/impl/im2row_gpu.cu
cpp_src+=matlab/src/bits/impl/subsample_gpu.cu
cpp_src+=matlab/src/bits/impl/copy_gpu.cu
cpp_src+=matlab/src/bits/impl/pooling_gpu.cu
cpp_src+=matlab/src/bits/impl/normalize_gpu.cu
cpp_src+=matlab/src/bits/datacu.cu
endif

# cuDNN-specific files
ifneq ($(ENABLE_CUDNN),)
cpp_src+=matlab/src/bits/impl/nnconv_cudnn.cu
cpp_src+=matlab/src/bits/impl/nnpooling_cudnn.cu
endif

# Other files
ifneq ($(ENABLE_IMREADJPEG),)
mex_src+=matlab/src/vl_imreadjpeg.c
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
	mkdir -p $(*)/ ; touch $(*)/.stamp
$(mex_tgt): matlab/mex/.build/impl/.stamp
$(cpp_tgt): matlab/mex/.build/impl/.stamp

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
