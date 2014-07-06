# file: Makefile
# author: Andrea Vedaldi
# brief: matconvnet makefile for mex files

# Copyright (C) 2007-14 Andrea Vedaldi and Brian Fulkerson.
# All rights reserved.
#
# This file is part of the VLFeat library and is made available under
# the terms of the BSD license (see the COPYING file).

ENABLE_GPU ?=
NVCC ?= /Developer/NVIDIA/CUDA-6.0/bin/nvcc
MATLABROOT ?= /Applications/MATLAB_R2014a.app

SHELL = /bin/bash # sh not good enough

# at least compute 2.0 required
NVCCOPTS = -gencode=arch=compute_20,code=sm_21 -gencode=arch=compute_30,code=sm_30

MEX = $(MATLABROOT)/bin/mex
MEXEXT = $(MATLABROOT)/bin/mexext
MEXARCH = $(subst mex,,$(shell $(MEXEXT)))
MEXOPTS =  -largeArrayDims -lmwblas
MEXOPTS_GPU = \
-DENABLE_GPU -f matlab/src/config/mex_CUDA_$(MEXARCH).xml \
-largeArrayDims -lmwblas -lcudart -lcublas \
-v

ifneq ($(DEBUG),)
MEXOPTS += -g
NVCCOPTS += -g
endif

.PRECIOUS: matlab/src/bits/pooling_gpu.o matlab/src/bits/im2col.o matlab/src/bits/normalize.o matlab/src/bits/im2col_gpu.o matlab/src/bits/pooling.o matlab/src/bits/normalize_gpu.o

# --------------------------------------------------------------------
#                                                           Do the job
# --------------------------------------------------------------------

nvcc_filter=2> >( sed 's/^\(.*\)(\([0-9][0-9]*\)): \([ew].*\)/\1:\2: \3/g' >&2 )

cpp_src:=matlab/src/bits/im2col.cpp
cpp_src+=matlab/src/bits/pooling.cpp
cpp_src+=matlab/src/bits/normalize.cpp

ifeq ($(ENABLE_GPU),)
mex_src:=matlab/src/vl_nnconv.cpp
mex_src+=matlab/src/vl_nnpool.cpp
mex_src+=matlab/src/vl_nnnormalize.cpp
else
mex_src:=matlab/src/vl_nnconv.cu
mex_src+=matlab/src/vl_nnpool.cu
mex_src+=matlab/src/vl_nnnormalize.cu
cpp_src+=matlab/src/bits/im2col_gpu.cu
cpp_src+=matlab/src/bits/pooling_gpu.cu
cpp_src+=matlab/src/bits/normalize_gpu.cu
endif

mex_tgt:=$(subst matlab/src/,matlab/mex/,$(mex_src))
mex_tgt:=$(patsubst %.cpp,%.mex$(MEXARCH),$(mex_tgt))
mex_tgt:=$(patsubst %.cu,%.mex$(MEXARCH),$(mex_tgt))

cpp_tgt:=$(patsubst %.cpp,%.o,$(cpp_src))
cpp_tgt:=$(patsubst %.cu,%.o,$(cpp_tgt))

.PHONY: all, distclean, clean, info

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
	   $(MEXOPTS_GPU) "$(<)" \
	   -output "$(@)" $(cpp_tgt) $(nvcc_filter)
endif

# Other targets
info:
	@echo "mex_src=$(mex_src)"
	@echo "mex_tgt=$(mex_tgt)"
	@echo "cpp_src=$(cpp_src)"
	@echo "cpp_tgt=$(cpp_tgt)"

clean:
	find . -name '*~' -delete
	rm -f $(cpp_tgt)
	rm -rf matlab/mex/

distclean: clean
	rm -rf matlab/mex/
