# MatConvNet: CNNs for MATLAB

Version 1.0-beta2.

**MatConvNet** is a MATLAB toolbox implementing *Convolutional Neural
Networks* (CNNs) for computer vision applications. It is simple,
efficient, and can learn and run state-of-the-art CNNs. Several
example CNNs are included to classify and encode images. See
[here](#about) for further details.

- Tarball for [version 1.0-beta2](download/matconvnet-1.0-beta2.tar.gz)
- [GIT repository](http://www.github.com/vlfeat/matconvnet.git)
- [PDF manual](matconvnet-manual.pdf) (see also MATLAB inline help).
- [Installation instructions](#installing)

**Contents**

- [Getting started](#started)
  - [Using pre-trained models](#pretrained)
  - [Training your own models](#training)
  - [Working with GPU accelerated code](#gpu)
- [Installing](#installing)
  - [Compiling](#compiling)
- [About MatConvNet](#about)

## <a name='started'></a> Getting started

This section provides a practical introduction to this toolbox.
Please see the [reference PDF manual](matconvnet-manual.pdf) for
technical details. MatConvNet can be used to
[train CNN models](#training), or it can be used to evaluate several
[pre-trained models](#pretrained) on your data.

### <a name='pretrained'></span> Using pre-trained models

This section describes how pre-trained models can be downloaded and
used in MatConvNet. The following models are available

- VGG models from the
  [Return of the Devil](http://www.robots.ox.ac.uk/~vgg/research/deep_eval) paper:
  - [imagenet-vgg-f](models/imagenet-vgg-f.mat)
  - [imagenet-vgg-m](models/imagenet-vgg-m.mat)
  - [imagenet-vgg-s](models/imagenet-vgg-s.mat)
  - [imagenet-vgg-m-2048](models/imagenet-vgg-m-2048.mat)
  - [imagenet-vgg-m-1024](models/imagenet-vgg-m-1024.mat)
  - [imagenet-vgg-m-128](models/imagenet-vgg-m-128.mat)
- Berkeley [Caffe reference models](http://caffe.berkeleyvision.org/getting_pretrained_models.html):
  - [imagenet-caffe-ref](models/imagenet-caffe-ref.mat)
  - [imagenet-caffe-alex](models/imagenet-caffe-alex.mat)

For example, in order to run ImageNet-S on a test image, use:

    % setup MtConvNet in MATLAB
    run matlab/vl_setupnn

    % download a pre-trained CNN from the web
    gunzip('http://www.vlfeat.org/sandbox-matconvnet/models/imagenet-vgg-f.mat') ;
    net = load('imagenet-vgg-f.mat') ;

    % obtain and preprocess an image
    im = imread('peppers.png') ;
    im_ = single(im) ; % note: 255 range
    im_ = imresize(im_, net.normalization.imageSize(1:2)) ;
    im_ = im_ - net.normalization.averageImage ;

    % run the CNN
    res = vl_simplenn(net, im_) ;

    % show the classification result
    scores = squeeze(gather(res(end).x)) ;
    [bestScore, best] = max(scores) ;
    figure(1) ; clf ; imagesc(im) ;
    title(sprintf('%s (%d), score %.3f',...
    net.classes.description{best}, best, bestScore)) ;

`vl_simplenn` is a wrapper around MatConvNet core computational blocks
implements a CNN with a simple linear topology (a chain of layers). It
is not needed to use the toolbox, but it simplifies common examples
such as the ones discussed here. See also
`examples/cnn_imagenet_minimal.m` for further examples.

### <a name='training'></a> Training your own models

MatConvNet can be used to train models using stochastic gradient
descent and back-propagation.

The following learning demos are provided:

- **MNIST**. See `examples/cnn_mnist.m`.
- **CIFAR**. See `examples/cnn_cifar.m`.
- **ImageNet**. See `examples/cnn_imagenet.m`.

These are self-contained, and include downloading and unpacking of the
data. MNIST and CIFAR are small datasets (by today's standard) and
training is feasible on a CPU. For ImageNet, however, a powerful GPU
is highly recommended.

###  <a name='gpu'></a> Working with GPU acelereated code

GPU support in MatConvNet relies on the corresponding support as
provided by recent releases of MATLAB and of its Parallel Programming
Toolbox. This toolbox relies on CUDA-compatible cards, and you will
need a copy of the CUDA devkit to compile GPU support in MatConvNet
(see above).

All the core computational functions (e.g. `vl_nnconv`) in the toolbox
can work with either MATLAB arrays or MATLAB GPU arrays. Therefore,
switching to use the a GPU is as simple as converting the input CPU
arrays in GPU arrays.

In order to make the very best of powerful GPUs, it is important to
balance the load between CPU and GPU in order to avoid starving the
latter. In training on a problem like ImageNet, the CPU(s) in your
system will be busy loading data from disk and streaming it to the GPU
to evaluate the CNN and its derivative. MatConvNet includes the
utility `vl_imreadjpeg` to accelerate and parallelize loading images
into memory (this function is currently a bottleneck will be made more
powerful in future releases).

## <a name='installing'></a> Installation

This library comprises several MEX files that need to be compiled
before MATLAB can use it. Start by downloading and unpacking the code;
then follow at the [compilation](#compiling) instructions to compile
the MEX file. Once the MEX files are properly compiled, MATLAB setup
is easy. Simply start MATLAB and type

    > run <path to MatConvNet>/matlab/vl_setupnn

At this point the library should be ready to use. To test it, try
issuing the command:

    > vl_test_nnlayers

### <a name='compiling'></a> Compiling

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
you will need a corresponding version of the CUDA (e.g. CUDA-5.5 for
R2014a) -- use the `gpuDevice` MATLAB command to figure out the proper
version of the CUDA toolkit. Then

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

## <a name='about'></a> About MatConvNet

MatConvNet was born in the Oxford Visual Geometry Group as both an
educatinonal and research platform for fast prototyping in
Convolutional Neural Nets. Its main features are:

- *Flexibility.* Neural network layers are implemented in a
  straightforward manner, often directly in MATLAB code, so that they
  are easy to modify, extend, or integrate with new ones. Other
  toolboxes hide the neural network layers behind a wall of compiled
  code; here the granularity is much finer.
- *Power.* The implementation can run large models such as Krizhevsky
  et al., including the DeCAF and Caffe variants. Several pre-trained
  models are provided.
- *Efficiency.* The implementation is quite efficient, supporting both
  CPU and GPU computation.

This library may be merged in the future with
[VLFeat library](http://www.vlfeat.org/). It uses a very similar
style, so if you are familiar with VLFeat, you should be right at home
here.

### Copyright and acknowledgments

This package was created and is currently maintained by
[Andrea Vedaldi](http://www.robots.ox.ac.uk/~vedaldi) and Karel
Lenc. It is distributed under the permissive BSD license (see the file
`COPYING`).

    Copyright (c) 2014 Andrea Vedaldi and Karel Lenc.
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

The implementation of the computational blocks in this library, and in
particular of the convolution operator, is inspired by
[Caffe](http://caffe.berkeleyvision.org).

## Changes

- 1.0-beta2 (June 2014) Adds a set of standard models.
- 1.0-beta1 (June 2014) First public release
