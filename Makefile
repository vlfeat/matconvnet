MEX=/Applications/MATLAB_R2013a.app/bin/mex
SHELL=/bin/bash
NVCC=/Developer/NVIDIA/CUDA-5.5/bin/nvcc
MEXARCH=maci64
MEXOPTS=-f src/mex_gpu_opts.sh -lmwblas -lcudart -lcublas -largeArrayDims
ifneq ($(DEBUG),)
MEXOPTS+=-g
endif

cus=$(wildcard src/*.cu)
mexs=$(subst src/,mex/,$(patsubst %.cu,%.mex$(MEXARCH),$(cus)))

.PHONY: all, distclean, clean, info

all: $(mexs)

mex/.stamp:
	mkdir mex ; touch mex/.stamp

# The horrible filter below makes NVCC generate errors in a format
# compatbile with Xcode
mex/%.mexmaci64 : src/%.cu mex/.stamp
	MW_NVCC_PATH='$(NVCC)' \
	$(MEX) $(MEXOPTS) "$(<)" -o "$(@)" \
	 2> >(sed 's/^\(.*\)(\([0-9][0-9]*\)): \([ew].*\)/\1:\2: \3/g' >&2)

mex/gconv.mexmaci64 : $(shell echo src/bits/im2col.{cpp,hpp})
mex/gpool.mexmaci64 : $(shell echo src/bits/pooling.{cpp,hpp})

info:
	@echo "cus=$(cus)"
	@echo "mexs=$(mexs)"

clean:
	rm -rf mex/

distclean: clean
