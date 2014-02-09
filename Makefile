MEX=/Applications/MATLAB_R2013a.app/bin/mex

all: fast_conv.mexmaci64

%.mexmaci64 : %.cu
	MW_NVCC_PATH='/Developer/NVIDIA/CUDA-5.5/bin/nvcc' \
	$(MEX) -f mex_gpu_opts.sh -lcudart -lcublas \
	  $(<) -o $(@) -largeArrayDims
