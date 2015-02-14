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
<a name='compiling'></a>

MatConvNet compiles under Linux, Mac, and Windows (with the exception
of the `vl_imreadjpeg` tool which is not yet supported under
Windows). This page discusses compiling MatConvNet using the MATLAB
function `vl_compilenn`. While this is the easiest method,
[alternative compilation methods](install-alt.md) are possible and may
be more conveniente for a library developer.

### Compiling MatConvNet

If this is the first time you compile MatConvNet, consider trying
first to compile the CPU only version. Compiling the library is
obtained by using the [`vl_compilenn`](mfiles/vl_compilenn)
command. Follow these steps:

1.  Make sure that MATLAB is
    [configured to use your compiler](http://www.mathworks.com/help/matlab/matlab_external/changing-default-compiler.html).
2.  Unpack MatConvNet in a location of your choice. Call this
    location `<MatConvNet>`.
3.  Open MATLAB and change the current directory to the copy of
    MatConvNet just created. From MATLAB's prompt:

        > cd <MatConvNet>

4.  At MALATB's prompt, issue the commands:

        > addpath matlab
        > vl_compilenn

At this point MatConvNet should start compiling. If all goes well, you
are ready to use the library. If not, you can try debugging the
problem by running the complation script again in verbose mode:

    > vl_compilenn('verbose', 1)

Increase the verbosity level to 2 to get even more information.

### Compiling the GPU support

To use the GPU code, you will also need a NVIDA GPU card with compute
capability 2.0 or greater and a copy of the NVIDIA CUDA toolkit
**corresponding to your MATLAB version** (see the next section for an
alternative):

| MATLAB    | CUDA toolkit      |
|-----------|-------------------|
| R2013b    | 5.5               |
| R2014a    | 6.0               |
| R2014b    | 6.5               |

You can also use the `gpuDevice` MATLAB command to find out the
correct version of the CUDA toolkit. Assuming that there is only a
single copy of the CUDA toolkit installed in your system and that it
matches MATLAB's version, simply use:

    > vl_compilenn('enableGpu', true)

If you have multiple versions of the CUDA toolkit, or if the script
cannot find the toolkit for any reason, specify the latter
directly. For example, on a Mac this may look like:

    > vl_compilenn('enableGpu', true, 'cudaRoot', '/Developer/NVIDIA/CUDA-6.0')

Once more, you can use the `verbose` option to obtain more information
if needed.

### Using an unsupported CUDA toolkit version

MatConvNet can be compiled to use a more recent version fo the CUDA
toolkit than the one officially supported. While this may cause
unforseen issues (although none is known so far), it is necessary to
use recent libraries such as [CuDNN]().

Compiling with a newer version of CUDA requires using the
`cudaMethod,nvcc` option. For example, on a Mac this may look like:

    > vl_compilenn('enableGpu', true, ...
                   'cudaRoot', '/Developer/NVIDIA/CUDA-6.5', ...
                   'cudaMethod', 'nvcc')

Note that at this point MatConvNet MEX files are linked *against the
system CUDA libraries* instead of the one distributed with
MATLAB. Hence, in order to use MatConvNet it is now necessary to allow
MATLAB accessing the corresponding libraries. On Linux and Mac, one
way to do so is to start MATLAB from the command line (terminal)
specifying the `LD_LIBRARY_PATH` option. For instance, on a Mac this
may look like:

    $ LD_LIBRARY_PATH=/Developer/NVIDIA/CUDA-6.5/lib /Applications/MATLAB_R2014b.app/bin/matlab

### Compiling the CuDNN support

MatConvNet supports the NVIDIA CuDNN library for deep learning (and in
particular their fast convolution code). In order to use it, obtain
the
[CuDNN Candidate Release 2](http://devblogs.nvidia.com/parallelforall/accelerate-machine-learning-cudnn-deep-neural-network-library). Note
that only Candidate Release 2 has been tested so far (Candidate
Release 1 will *not* work). Make sure that the CUDA toolkit matches
the one in CuDNN (e.g. 6.5). This often means that the CUDA toolkit
will *not* match the one used internally by MATLAB.

Unpack the CuDNN library binaries and header files in a place of you
choice. In the rest of the instructions, it will be assumed that this
is a new directory called `local/` in the `<MatConvNet>` root
directory.

Use `vl_compilenn` with the `cudnnEnable,true` option to compile the
library; do not forget to use `cudaMethod,nvcc` as, at it is likely,
the CUDA toolkit version is newer than MATLAB's CUDA toolkit. For
example, on Mac this may look like:

    > vl_compilenn('enableGpu', true, ...
                   'cudaRoot', '/Developer/NVIDIA/CUDA-6.5', ...
                   'cudaMethod', 'nvcc', ...
                   'enableCudnn', 'true', ...
                   'cudnnRoot', 'local/') ;

MatConvNet is now compiled with CuDNN support. When starting MATLAB,
however, do not forget to point it to the paths of both the CUDA
libraries as well as the CuDNN ones. On a Mac terminal, this may look
like:

    $ cd <MatConvNet>
    $ LD_LIBRARY_PATH=/Developer/NVIDIA/CUDA-6.5/lib:local /Applications/MATLAB_R2014b.app/bin/matlab

### Compiling `vl_imreadjpeg`
<a name='jpeg'></a>

The `vl_imreadjpeg` function in the MatConvNet toolbox accelerates
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

If LibJPEG is installed elsewhere, you would have to replace the paths
`/opt/local/include` and `/opt/local/lib` accordingly.

### Further examples

To compile all the features in MatConvNet on a Mac and MATLAB 2014b,
CUDA toolkit 6.5 and CuDNN Release Candidate 2, use:

    > vl_compilenn('enableGpu', true, ...
                   'enableCudnn', true, ...
                   'cudaMethod', 'nvcc', ...
                   'cudnnRoot', 'local/', ...
                   'cudaRoot', '/Developer/NVIDIA/CUDA-6.5', ...
                   'enableImreadJpeg', true,  ...
                   'imreadJpegCompileFlags', {'-I/opt/local/include'}, ...
                   'imreadJpegLinkFlags', {'-L/opt/local/lib','-ljpeg'}) ;

