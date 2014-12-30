# Installing and compiling the library

This library comprises several MEX files that need to be compiled
before MATLAB can use it. Start by downloading and unpacking the code;
then follow at the [compilation](#compiling) instructions to compile
the MEX file. Once the MEX files are properly compiled, MATLAB setup
is easy. Simply start MATLAB and type

    > run <path to MatConvNet>/matlab/vl_setupnn

At this point the library should be ready to use. To test it, try
issuing the command:

    > vl_test_nnlayers

To test the GPU version of the library (provided that you have a GPU
and have compiled the corresponding support), use

    > vl_test_nnlayers(true)

Note that this is actually slower than the CPU version; this is
expected and an artifact of the test code.

## Compiling

<a name='compiling'/></a>

### Compiling from MATLAB

Make sure that you have a C++ compiler configured in MATLAB (see `mex
-setup`). Then the simplest method to compile the library is to use
the provided [`vl_compilenn`](mfiles/vl_compilenn) command:

    > run <path to MatConvNet>/matlab/vl_setupnn
    > vl_compilenn()

Read the [function documentation](mfiles/vl_compilenn) for further
information on the options.

To compile the GPU code, you will also need a copy of the NVIDIA CUDA
Devkit, preferably corresponding to your MATLAB version, and a NVIDA
GPU with compute capability 2.0 or greater. Then

    > vl_compilenn('enableGpu', true)

should do the trick.

If you want to run experiments on ImageNet or similar large scale
dataset, the `vl_imreadjpeg` function is also needed. This requires a
copy of LibJPEG to be installed in your system and usable by the C++
compiler. To compile this file use:

    > vl_compilenn('enableGpu', true, 'enableImreadJpeg', true)

At present, this function is supported only under Mac and Linux.

### Compiling from the Shell

This method works only for Mac and Linux and uses the supplied
`Makefile`:

    > make ARCH=<your arch> MATLABROOT=<path to MATLAB>

This requires MATLAB to be correctly configured with a suitable
compiler (usually Xcode for Mac, GCC for Linux, Visual C for Windows).
For example:

    > make ARCH=maci64 MATLABROOT=/Applications/MATLAB_R2014a.app

should work for a Mac with MATLAB R2014 installed in its default
location. The other supported architecture is `glnxa64` (for Linux).

Compiling the GPU version requries some more configuration. First of
all, you will need a recent version of MATLAB (e.g. R2014a). Secondly,
you will need a corresponding version of the
[CUDA toolkit](https://developer.nvidia.com/cuda-toolkit-archive)
(e.g. CUDA-5.5 for R2014a) -- use the `gpuDevice` MATLAB command to
figure out the proper version of the CUDA toolkit. Then

    > make ENABLE_GPU=y ARCH=<your arch> MATLABROOT=<path to MATLAB> CUDAROOT=<path to CUDA>

should do the trick. For example:

    > make ENABLE_GPU=y ARCH=maci64 MATLABROOT=/Applications/MATLAB_R2014a.app CUDAROOT=/Developer/NVIDIA/CUDA-5.5

should work on a Mac with MATLAB R2014a.

Finally, running large-scale experiments on fast GPUs require reading
and preprocessing JPEG images very efficiently. To this end,
MatConvNet ships with a `vl_imreadjpeg` tool that can be used to read
JPEG images in a separate thread. This tool is Linux/Mac only and
requires LibJPEG and POSIX threads. Compile it by switching on the
`ENABLE_IMREADJPEG` flag:

    > make ENABLE_IMREADJPEG=y
