MEX=/Applications/MATLAB_R2013a.app/bin/mex
SHELL=/bin/bash
NVCC=/Developer/NVIDIA/CUDA-5.5/bin/nvcc
MEXARCH=maci64
MEXOPTS=-f matlab/src/mex_gpu_opts.sh -lmwblas -lcudart -lcublas -largeArrayDims -v
ifneq ($(DEBUG),)
MEXOPTS+=-g
endif

cus=$(wildcard matlab/src/*.cu)
mexs=$(subst matlab/src/,matlab/mex/,$(patsubst %.cu,%.mex$(MEXARCH),$(cus)))

.PHONY: all, distclean, clean, info

all: $(mexs)

matlab/mex/.stamp:
	mkdir matlab/mex ; touch matlab/mex/.stamp

# The horrible filter below makes NVCC generate errors in a format
# compatbile with Xcode
matlab/mex/%.mexmaci64 : matlab/src/%.cu matlab/mex/.stamp
	MW_NVCC_PATH='$(NVCC)' \
	$(MEX) $(MEXOPTS) "$(<)" -o "$(@)" \
	 2> >(sed 's/^\(.*\)(\([0-9][0-9]*\)): \([ew].*\)/\1:\2: \3/g' >&2)

matlab/mex/gconv.mexmaci64 : $(shell echo matlab/src/bits/im2col.{cpp,hpp})
matlab/mex/gpool.mexmaci64 : $(shell echo matlab/src/bits/pooling.{cpp,hpp})
matlab/mex/gnormalize.mexmaci64 : $(shell echo matlab/src/bits/normalize.{cpp,hpp})

info:
	@echo "cus=$(cus)"
	@echo "mexs=$(mexs)"

clean:
	find . -name '*~' -delete

distclean: clean
	rm -rf matlab/mex/
