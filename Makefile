MEX=/Applications/MATLAB_R2013a.app/bin/mex

NVCC=/Developer/NVIDIA/CUDA-5.5/bin/nvcc
MEXARCH=maci64
MEXOPTS=-f src/mex_gpu_opts.sh -lcudart -lcublas -largeArrayDims

cus=$(wildcard src/*.cu)
mexs=$(subst src/,mex/,$(patsubst %.cu,%.mex$(MEXARCH),$(cus)))

.PHONY: all, distclean, clean, info

all: $(mexs)

mex/.stamp:
	mkdir mex ; touch mex/.stamp

mex/%.mexmaci64 : src/%.cu mex/.stamp
	MW_NVCC_PATH='$(NVCC)' \
	$(MEX) $(MEXOPTS) "$(<)" -o "$(@)"

info:
	@echo "cus=$(cus)"
	@echo "mexs=$(mexs)"

clean:
	rm -rf mex/

distclean: clean
