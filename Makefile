# file: Makefile
# author: Andrea Vedaldi
# brief: matconvnet makefile for mex files

# Copyright (C) 2014-17 Andrea Vedaldi.
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
ENABLE_DOUBLE ?= yes

# MATLAB, CUDA, CUDNN paths

# Linux
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
ARCH ?= glnxa64
MATLABROOT ?= $(shell readlink -f `which matlab` | rev | cut -d'/' -f3- | rev)
CUDAROOT ?= /usr/local/cuda
endif

# macOS
ARCH ?= maci64
MATLABROOT ?= /Applications/MATLAB_R2017a.app
CUDAROOT ?= /Developer/NVIDIA/CUDA-8.0
# Remark: each MATLAB version requires a particular CUDA Toolkit version.
# Note that multiple CUDA Toolkits can be installed.
#MATLABROOT ?= /Applications/MATLAB_R2014b.app
#CUDAROOT ?= /Developer/NVIDIA/CUDA-6.0
#MATLABROOT ?= /Applications/MATLAB_R2015a.app
#CUDAROOT ?= /Developer/NVIDIA/CUDA-7.0
#MATLABROOT ?= /Applications/MATLAB_R2015b.app
#CUDAROOT ?= /Developer/NVIDIA/CUDA-7.5

CUDNNROOT ?= "$(CURDIR)/local/"
CUDAMETHOD ?= $(if $(ENABLE_CUDNN),nvcc,mex)

# For Mac OS X: Use this to use an old Xcode (for CUDA) after installing
# the corresponding Xcode Command Line Tools from developer.apple.compile
# DEVELOPER_DIR=/Applications/Xcode7.3.1.app/Contents/Developer


# Maintenance
NAME = matconvnet
VER = 1.0-beta25
DIST = $(NAME)-$(VER)
LATEST = $(NAME)-latest
RSYNC = rsync
HOST = vlfeat-admin:sites/sandbox-matconvnet
GIT = git
SHELL = /bin/bash # sh not good enough

# --------------------------------------------------------------------
#                                                        Configuration
# --------------------------------------------------------------------

# General options
MEX = $(MATLABROOT)/bin/mex
MEXEXT = $(MATLABROOT)/bin/mexext
MEXARCH = $(subst mex,,$(shell $(MEXEXT)))
MEXOPTS ?= matlab/src/config/mex_CUDA_$(ARCH).xml
NVCC = $(CUDAROOT)/bin/nvcc

comma:=,
space:=
space+=
join-with = $(subst $(space),$1,$(strip $2))
nvcc_quote = $(if $(strip $1),-Xcompiler $(call join-with,$(comma),$(1)),)
nvcc_filter := 2> >( sed 's/^\(.*\)(\([0-9][0-9]*\)): \([ew].*\)/\1:\2: \3/g' >&2 )
nvcc_filter :=

# BASEFLAGS: Base flags passed to `mex` and `nvcc` always.
BASEFLAGS = \
$(if $(ENABLE_GPU),-DENABLE_GPU,) \
$(if $(ENABLE_DOUBLE),-DENABLE_DOUBLE,) \
$(if $(ENABLE_CUDNN),-DENABLE_CUDNN -I$(CUDNNROOT)/include,) \
$(if $(VERB),-v,) \
$(if $(DEBUG),-g -DDEBUG,-O -DNDEBUG)

# MEXFLAGS: Additional flags passed to `mex` for compiling C++
# code. The MEX_CXXCFLAGS options are passed directly to the
# encapsulated C++ compiler.
MEXFLAGS = -largeArrayDims \
CXXFLAGS='$$CXXFLAGS $(MEX_CXXFLAGS)' \
CXXOPTIMFLAGS='$$CXXOPTIMFLAGS $(MEX_CXXOPTIMFLAGS)'
ifdef MEXCONFIG
MEXFLAGS += -f "$(MEXCONFIG)"
endif

# MEXCUDA_FLAGS: Additional flags passed to `mex` for compiling C++
# code. The MEXCUDA_CXXFLAGS and MEXCUDA_CXXOPTIMFLAGS options are
# passed directly to the encapsualted `nvcc` compiler.
MEXCUDA_FLAGS = -largeArrayDims \
CXXFLAGS='$$CXXFLAGS $(MEXCUDA_CXXFLAGS)' \
CXXOPTIMFLAGS='$$CXXOPTIMFLAGS $(MEXCUDA_CXXOPTIMFLAGS)'
ifdef MEXCUDACONFIG
MEXCUDA_FLAGS += -f "$(MEXCUDACONFIG)"
endif

# MEXLINK_FLAGS: Aditional flags passed to `mex` for linking. The
# MEXLINK_LDFLAGS, MEXLINK_LDOPTIMFLAGS, and MEXLINK_LINKLIBS options
# are passed directly to the encapsulated C++ compiler/linker.
MEXLINK_FLAGS = -largeArrayDims \
LDFLAGS='$$LDFLAGS $(MEXLINK_LDFLAGS)' \
LDOPTIMFLAGS='$$LDOPTIMFLAGS $(MEXLINK_LDOPTIMFLAGS)' \
LINKLIBS='$(MEXLINK_LINKLIBS) $$LINKLIBS' \
-lmwblas

# Additional flags passed to `nvcc` for compiling CUDA code.
NVCCFLAGS = -D_FORCE_INLINES --std=c++11 --compiler-options=-fPIC \
-I"$(MATLABROOT)/extern/include" \
-I"$(MATLABROOT)/toolbox/distcomp/gpu/extern/include" \
-gencode=arch=compute_30,code=\"sm_30,compute_30\"

# --------------------------------------------------------------------
# Generic configuration
# --------------------------------------------------------------------

MEX_CXXFLAGS = --std=c++11

ifndef DEBUG
MEX_CXXOPTIMFLAGS += -mssse3 -ffast-math
MEXCUDA_CXXOPTIMFLAGS += --compiler-options=-mssse3,-ffast-math
NVCCFLAGS += --compiler-options=-mssse3,-ffast-math
endif

# --------------------------------------------------------------------
# Mac OS X
# --------------------------------------------------------------------
ifeq "$(ARCH)" "$(filter $(ARCH),maci64)"
IMAGELIB ?= $(if $(ENABLE_IMREADJPEG),quartz,none)
NVCCFLAGS += --compiler-options=-mmacosx-version-min=10.10
ifdef DEVELOPER_DIR
clang := $(shell DEVELOPER_DIR="$(DEVELOPER_DIR)" xcrun -f clang++)
NVCCFLAGS += --compiler-bindir="$(clang)"
#MEXCUDA_CXXFLAGS += --compiler-bindir="$(clang)"
endif
MEXLINK_FLAGS += \
$(if $(ENABLE_GPU),-L"$(CUDAROOT)/lib" -lmwgpu -lcudart -lcublas) \
$(if $(ENABLE_CUDNN),-L"$(CUDNNROOT)/lib" -lcudnn)
# rpath directive need bypass
MEXLINK_LDFLAGS += \
$(if $(ENABLE_GPU),-Wl$(comma)-rpath -Wl$(comma)"$(CUDAROOT)/lib") \
$(if $(ENABLE_CUDNN),-Wl$(comma)-rpath -Wl$(comma)"$(CUDNNROOT)/lib")
endif

# --------------------------------------------------------------------
# Linux
# --------------------------------------------------------------------
ifeq "$(ARCH)" "$(filter $(ARCH),glnxa64)"
IMAGELIB ?= $(if $(ENABLE_IMREADJPEG),libjpeg,none)
MEXLINK_FLAGS += \
-lrt \
$(if $(ENABLE_GPU),-L"$(CUDAROOT)/lib64" -lmwgpu -lcudart -lcublas) \
$(if $(ENABLE_CUDNN),-L"$(CUDNNROOT)/lib64" -lcudnn)
MEXLINK_LDFLAGS += \
$(if $(ENABLE_GPU),-Wl$(comma)-rpath -Wl$(comma)"$(CUDAROOT)/lib64") \
$(if $(ENABLE_CUDNN),-Wl$(comma)-rpath -Wl$(comma)"$(CUDNNROOT)/lib64")
endif

# --------------------------------------------------------------------
# Image library
# --------------------------------------------------------------------
ifeq ($(IMAGELIB),libjpeg)
MEXLINK_FLAGS += -ljpeg
endif
ifeq ($(IMAGELIB),quartz)
MEXLINK_LINKLIBS += -framework Cocoa -framework ImageIO
endif

# --------------------------------------------------------------------
#                                                      Build MEX files
# --------------------------------------------------------------------

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
cpp_src+=matlab/src/bits/nnnormalizelp.$(ext)
cpp_src+=matlab/src/bits/nnbnorm.$(ext)
cpp_src+=matlab/src/bits/nnbilinearsampler.$(ext)
cpp_src+=matlab/src/bits/nnroipooling.$(ext)
mex_src+=matlab/src/vl_nnconv.$(ext)
mex_src+=matlab/src/vl_nnconvt.$(ext)
mex_src+=matlab/src/vl_nnpool.$(ext)
mex_src+=matlab/src/vl_nnnormalize.$(ext)
mex_src+=matlab/src/vl_nnnormalizelp.$(ext)
mex_src+=matlab/src/vl_nnbnorm.$(ext)
mex_src+=matlab/src/vl_nnbilinearsampler.$(ext)
mex_src+=matlab/src/vl_nnroipool.$(ext)
mex_src+=matlab/src/vl_taccummex.$(ext)
mex_src+=matlab/src/vl_tmove.$(ext)
ifdef ENABLE_IMREADJPEG
mex_src+=matlab/src/vl_imreadjpeg.$(ext)
mex_src+=matlab/src/vl_imreadjpeg_old.$(ext)
endif

# CPU-specific files
cpp_src+=matlab/src/bits/impl/im2row_cpu.cpp
cpp_src+=matlab/src/bits/impl/copy_cpu.cpp
cpp_src+=matlab/src/bits/impl/tinythread.cpp
ifdef ENABLE_IMREADJPEG
cpp_src+=matlab/src/bits/impl/imread_$(IMAGELIB).cpp
cpp_src+=matlab/src/bits/imread.cpp
endif

# GPU-specific files
ifdef ENABLE_GPU
cpp_src+=matlab/src/bits/impl/im2row_gpu.cu
cpp_src+=matlab/src/bits/impl/copy_gpu.cu
cpp_src+=matlab/src/bits/datacu.cu
mex_src+=matlab/src/vl_cudatool.cu
ifdef ENABLE_CUDNN
cpp_src+=
endif
endif

mex_tgt:=$(patsubst %.cpp,%.mex$(MEXARCH),$(mex_src))
mex_tgt:=$(patsubst %.cu,%.mex$(MEXARCH),$(mex_tgt))
mex_tgt:=$(subst matlab/src/,matlab/mex/,$(mex_tgt))

mex_obj:=$(patsubst %.cpp,%.o,$(mex_src))
mex_obj:=$(patsubst %.cu,%.o,$(mex_obj))
mex_obj:=$(subst matlab/src/,matlab/mex/.build/,$(mex_obj))

cpp_tgt:=$(patsubst %.cpp,%.o,$(cpp_src))
cpp_tgt:=$(patsubst %.cu,%.o,$(cpp_tgt))
cpp_tgt:=$(subst matlab/src/,matlab/mex/.build/,$(cpp_tgt))

.PHONY: all, distclean, clean, info, pack, post, post-doc, doc

all: $(cpp_tgt) $(mex_obj) $(mex_tgt)

# Create build directory
%/.stamp:
	mkdir -p $(*)/ ; touch $(*)/.stamp
$(mex_tgt): matlab/mex/.build/bits/impl/.stamp
$(cpp_tgt): matlab/mex/.build/bits/impl/.stamp

# Standard code
.PRECIOUS: matlab/mex/.build/%.o
.PRECIOUS: %/.stamp

matlab/mex/.build/bits/impl/imread.o : matlab/src/bits/impl/imread_helpers.hpp
matlab/mex/.build/bits/impl/imread_quartz.o : matlab/src/bits/impl/imread_helpers.hpp
matlab/mex/.build/bits/impl/imread_gdiplus.o : matlab/src/bits/impl/imread_helpers.hpp
matlab/mex/.build/bits/impl/imread_libjpeg.o : matlab/src/bits/impl/imread_helpers.hpp

# --------------------------------------------------------------------
#                                                    Compilation rules
# --------------------------------------------------------------------

ifneq ($(ENABLE_GPU),)
ifeq ($(CUDAMETHOD),mex)
matlab/mex/.build/%.o : matlab/src/%.cu matlab/mex/.build/.stamp
	MW_NVCC_PATH='$(NVCC)' \
	$(MEX) -c $(BASEFLAGS) $(MEXCUDA_FLAGS) "$(<)" $(nvcc_filter)
	mv -f "$(notdir $(@))" "$(@)"
else
matlab/mex/.build/%.o : matlab/src/%.cu matlab/mex/.build/.stamp
	$(NVCC) $(BASEFLAGS) $(NVCCFLAGS) "$(<)" -c -o "$(@)" $(nvcc_filter)
endif
endif

matlab/mex/.build/%.o : matlab/src/%.cpp matlab/src/%.cu matlab/mex/.build/.stamp
	$(MEX) -c $(BASEFLAGS) $(MEXFLAGS) "$(<)"
	mv -f "$(notdir $(@))" "$(@)"

matlab/mex/.build/%.o : matlab/src/%.cpp matlab/mex/.build/.stamp
	$(MEX) -c $(BASEFLAGS) $(MEXFLAGS) "$(<)"
	mv -f "$(notdir $(@))" "$(@)"

matlab/mex/%.mex$(MEXARCH) : matlab/mex/.build/%.o $(cpp_tgt)
	$(MEX) $(BASEFLAGS) $(MEXLINK_FLAGS) "$(<)" -output "$(@)" $(cpp_tgt)

# --------------------------------------------------------------------
#                                                        Documentation
# --------------------------------------------------------------------

include doc/Makefile

# --------------------------------------------------------------------
#                                                          Maintenance
# --------------------------------------------------------------------

info: doc-info
	@echo "mex_src=$(mex_src)"
	@echo "mex_obj=$(mex_obj)"
	@echo "mex_tgt=$(mex_tgt)"
	@echo "cpp_src=$(cpp_src)"
	@echo "cpp_tgt=$(cpp_tgt)"
	@echo '------------------------------'
	@echo 'CUDAMETHOD=$(CUDAMETHOD)'
	@echo 'CXXFLAGS=$(CXXFLAGS)'
	@echo 'CXXOPTIMFLAGS=$(CXXOPTIMFLAGS)'
	@echo 'LDFLAGS=$(LDFLAGS)'
	@echo 'LDOPTIMFLAGS=$(LDOPTIMFLAGS)'
	@echo 'LINKLIBS=$(LINKLIBS)'
	@echo '------------------------------'
	@echo 'MEXARCH=$(MEXARCH)'
	@echo 'MEXFLAGS=$(MEXFLAGS)'
	@echo 'MEXFLAGS_CC_CPU=$(MEXFLAGS_CC_CPU)'
	@echo 'MEXFLAGS_CC_GPU=$(MEXFLAGS_CC_GPU)'
	@echo 'MEXFLAGS_LD=$(MEXFLAGS_LD)'
	@echo '------------------------------'
	@echo 'NVCC=$(NVCC)'
	@echo 'NVCCVER=$(NVCCVER)'
	@echo 'NVCCVER_LT_70=$(NVCCVER_LT_70)'
	@echo 'NVCCFLAGS_PASS=$(NVCCFLAGS_PASS)'
	@echo 'NVCCFLAGS=$(NVCCFLAGS)'


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
	ln -sf $(DIST).tar.gz $(LATEST).tar.gz

post: pack
	$(RSYNC) -aP $(DIST).tar.gz $(LATEST).tar.gz $(HOST)/download/

post-models:
	$(RSYNC) -aP data/models/*.mat $(HOST)/models/

post-doc: doc
	$(RSYNC) -aP doc/matconvnet-manual.pdf $(HOST)/
	$(RSYNC) \
		--recursive \
		--perms \
	        --verbose \
	        --delete \
	        --exclude=download \
	        --exclude=models \
	        --exclude=matconvnet-manual.pdf \
	        --exclude=.htaccess doc/site/site/ $(HOST)/

.PHONY: model-md5
model-md5:
	cd data/models ; md5sum *.mat | xargs  printf '| %-33s| %-40s|\n'
