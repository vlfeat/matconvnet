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

## Compiling

<a name='compiling'/></a>

Compiling the CPU version of MatConvNet is simple (presently Linux and
Mac OS X are supported; Windows should work, up to some modifications
to `vl_imreadjpeg.c`).  The simplest compilation method is to use
supplied `Makefile`:

    > make ARCH=<your arch> MATLABROOT=<path to MATLAB>

This requires MATLAB to be correctly configured with a suitable
compiler (usually XCode for Mac, gcc for Linux, Visual C for Windows).
For example:

    > make ARCH=maci64 MATLABROOT=/Applications/MATLAB_R2014a.app

would work for a Mac with MATLAB R2014 installed in its default
folder. Other supported architectures are `glnxa64` (for Linux) and
`win64` (for Windows).

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
