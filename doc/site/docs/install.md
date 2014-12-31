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
expected and an artefact of the test code.

## Compiling

<a name='compiling'/></a>

Most of MatConvNet compiles under Linux, Mac, and Windows (with the
exception of the `vl_imreadjpeg` tool which for the moment is not
supported under Windows). There are two compilation methods: using the
`vl_compilenn` MATLAB command or using a Makefile. These are described
next.

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

Running large-scale experiments on fast GPUs require reading JPEG
images very efficiently. To this end, MatConvNet ships with the
`vl_imreadjpeg` tool that can be used to read batches of JPEG images
in a separate thread. This tool requires a copy of LibJPEG to be
installed in your system and be usable by the MATLAB MEX compiler. To
compile `vl_imreadjpeg` use:

    > vl_compilenn('enableGpu', true, 'enableImreadJpeg', true)

At present, this function is supported only under Mac and Linux. See
the [section below](#jpeg) for further details.

### Compiling from the Shell

This method works only for Mac and Linux and uses the supplied
`Makefile`:

    > make ARCH=<your arch> MATLABROOT=<path to MATLAB>

This requires MATLAB to be correctly configured with a suitable
compiler (usually Xcode for Mac and GCC for Linux). For example:

    > make ARCH=maci64 MATLABROOT=/Applications/MATLAB_R2014a.app

should work for a Mac with MATLAB R2014 installed in its default
location. The other supported architecture is `glnxa64` (for Linux).

Compiling the GPU version requires some more configuration. First of
all, you will need a recent version of MATLAB (e.g. R2014a). Secondly,
you will need a corresponding version of the
[CUDA toolkit](https://developer.nvidia.com/cuda-toolkit-archive)
(e.g. CUDA-5.5 for R2014a) -- use the `gpuDevice` MATLAB command to
figure out the proper version of the CUDA toolkit. Then

    > make ENABLE_GPU=y ARCH=<your arch> MATLABROOT=<path to MATLAB> CUDAROOT=<path to CUDA>

should do the trick. For example:

    > make ENABLE_GPU=y ARCH=maci64 MATLABROOT=/Applications/MATLAB_R2014a.app CUDAROOT=/Developer/NVIDIA/CUDA-5.5

should work on a Mac with MATLAB R2014a.

To compile `vl_imreadjpeg` use the `ENABLE_IMREADJPEG` flag:

    > make ENABLE_IMREADJPEG=y

This requires LibJPEG to be installed and available to the MEX
compiler, as explained below.

### Compiling `vl_imreadjpeg`

<a name='jpeg'></a>

The `vl_imreadjpeg` function in the MatConvNet toolkit accelerates
reading large batches of JPEG images. In order to compile it, a copy
of LibJPEG and of the corresponding header files must be available to
the MEX compiler used by MATLAB.

On *Linux*, it usually suffices to install the LibJPEG developer
package (for example `libjpeg-dev` on Ubuntu Linux). Then both
`vl_compilenn()` and the Makefile should work out of the box.

On *Mac OS X*, LibJPEG can be obtained for example by using
[MacPorts](http://www.macports.org):

    > sudo port install jpeg

This makes the library available as `/opt/local/lib/libjpeg.dylib` and
the header file as `/opt/local/include/jpeglib.h`. If you compile the
library us using `vl_compilenn()`, you can pass the location of these
files as part of the `ImreadJpegFlags` option:

    > vl_compilenn('enableImreadJpeg', true, 'imreadJpegFlags', ...
        {'-I/opt/local/include','-L/opt/local/lib','-ljpeg'});

If LibJPEG is installed elsewhere, you would have to modify
`/opt/local/include` and `/opt/local/lib` accordingly.

If you compile the library using the Makefile, edit the latter to
change the path to the LibJPEG library and header files
appropriately. In particular, search for the line

    matlab/mex/vl_imreadjpeg.mex$(MEXARCH): MEXFLAGS+=-I/opt/local/include -L/opt/local/lib -ljpeg

and change `/opt/local/include` and `/opt/local/lib` as required.
