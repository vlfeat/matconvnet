# MatConvNet: Convolutional Neural Networks for MATLAB

Version 1.0-beta.

**MatConvNet** is a simple MATLAB toolbox implementing Convolutional
Neural Networks (CNN) for computer vision applications. Its main
features are:

- *Flexibility.* Neural network layers are implemented in a
  straightforward manner, often directly in MATLAB code, so that they
  are easy to modify, extend, or integrate with new ones. Other
  toolboxes hide the whole neural network layers behind a wall of
  compiled code; here the granularity is much finer.
- *Power.* The implementation can run the latest features such as
  Krizhevsky et al., including the DeCAF and Caffe
  variants. Pre-learned features for different tasks can be easily
  downloaded.
- *Efficiency.* The implementation is quite efficient, supporting both
  CPU and GPU computation. Despite MATLAB overhead, it is only
  marginally slower than alternative implementations.

This library will be merged in the future with
[VLFeat](http://www.vlfeat.org/) library.

## Installation

This library comprises several MEX files that need to be compiled
before MATLAB can use it.

### Download

You can download a copy of the source code here:

- [Tarball]()
- [GIT repository](http://www.github.com/vlfeat/matconvnet.git)

### Compiling

Compiling the CPU version of MatConvNet is a simple affair. The simple
method is to use supplied `Makefile`:

    > make ARCH=<your arch> MATLABROOT=<path to MATLAB>

This requires MATLAB to be correctly configured with a suitable
compiler (usually XCode for Mac, gcc for Linux, Visual C for Windows).
For example:

    > make ARCH=maci64 MATLABROOT=/Applications/MATLAB_R2014a.app

would work for a Mac with MATLAB R2014 installed in its default
folder. Other supported architectures are `glnxa64` (for Linux) and
`win64` (for Windows).

Compiling the GPU version should be just as simple; however, in
practice, this may require some configuration. First of all, you will
need a recent version of MATLAB (e.g. R2014a). Secondly, you will need
a corresponding version of the CUDA (e.g. CUDA-5.5 for R2014a) -- use
the `gpuDevice` MATLAB command to figure out the proper version of the
CUDA toolkit. Then

    > make ENABLE_GPU=y ARCH=<your arch> MATLABROOT=<path to MATLAB> CUDAROOT=<path to CUDA>

should do the trick. For example:

    > make ENABLE_GPU=y ARCH=maci64 MATLABROOT=/Applications/MATLAB_R2014a.app CUDAROOT=/Developer/NVIDIA/CUDA-5.5

should work on a Mac with the corresponding version of MATLAB.

### Installing in MATLAB and testing the setup

Once the library is installed in MATLAB, setup is easy. Simply start MATLAB
and type

    > run <path to MatConvNet>/matlab/vl_setupnn

At this point the library should be ready to use. To test it, try

    > vl_test_nnlayers

## Usage

Please see the [reference PDF manual](matconvnet-manual.pdf) for technical details. There
are several examples provided for your convenience

### Pre-trained networks

You can download the following models

- [ImageNet-S](). A relatively small network pre-trained on
  ImageNet. Similar to Krizhevsky et al. 2012 network.
- [ImageNet-M](). A somewhat larger network, with better performance.
- [ImageNet-L](). A large model.

## About

This package was created and is currently maintained by
[Andrea Vedaldi](http://www.robots.ox.ac.uk/~vedaldi). It is
distributed under the permissive BSD license.

    Copyright (c) 2014 Andrea Vedaldi.
    All rights reserved.

    Redistribution and use in source and binary forms are permitted
    provided that the above copyright notice and this paragraph are
    duplicated in all such forms and that any documentation,
    advertising materials, and other materials related to such
    distribution and use acknowledge that the software was developed
    by the <organization>. The name of the <organization> may not be
    used to endorse or promote products derived from this software
    without specific prior written permission.  THIS SOFTWARE IS
    PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR IMPLIED WARRANTIES,
    INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTIES OF
    MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.

## Changes

- 1.0-beta. First public release (summer 2014).
