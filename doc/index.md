bet# MatConvNet: CNNs for MATLAB

**MatConvNet** is a MATLAB toolbox implementing *Convolutional Neural
Networks* (CNNs) for computer vision applications. It is simple,
efficient, and can run and learn state-of-the-art CNNs. Several
example CNNs are included to classify and encode images.

- [Homepage](http://www.vlfeat.org/matconvnet)
- Tarball for [version 1.0-beta7](download/matconvnet-1.0-beta7.tar.gz)
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

This section provides a practical introduction to MatConvNet. Please
see the [reference manual (PDF)](matconvnet-manual.pdf) for technical
details. MatConvNet can be used to evaluate several
[pre-trained CNNs](#pretrained) or to [train new models](#training).

### <a name='pretrained'></a> Using pre-trained models

This section describes how pre-trained models can be downloaded and
used in MatConvNet.

> **Remark:** The following CNN models have been *imported from other
> reference implementations* and are equivalent to the originals up to
> numerical precision. However, note that:
>
> 1. Images need to be pre-processed (resized and cropped) before
>    being submitted to a CNN for evaluation. Even small differences
>    in the prepreocessing details can have a non-negligible effect on
>    the results.
> 2. The example below shows how to evaluate a CNN, but does not
>    include data augmentation or encoding normalization as for
>    example provided by the
>    [VGG code](http://www.robots.ox.ac.uk/~vgg/research/deep_eval).
>    While this is easy to implement, it is not done automatically here.
> 3. These models are provided here for convenience, but please credit
>    the original authors.

- VGG models form the
  [Very Deep Convolutional Networks](http://www.robots.ox.ac.uk/~karen/)
  - **Citation:** `Very Deep Convolutional Networks for Large-Scale
  Image Recognition', *Karen Simonyan and Andrew Zisserman,* arXiv
  technical report, 2014, ([paper](http://arxiv.org/abs/1409.1556/)):
  - [imagenet-vgg-verydeep-16](models/imagenet-vgg-verydeep-16.mat)
  - [imagenet-vgg-verydeep-19](models/imagenet-vgg-verydeep-19.mat)

- VGG models from the
  [Return of the Devil](http://www.robots.ox.ac.uk/~vgg/research/deep_eval)
  paper (v1.0.1):
  - [imagenet-vgg-f](models/imagenet-vgg-f.mat)
  - [imagenet-vgg-m](models/imagenet-vgg-m.mat)
  - [imagenet-vgg-s](models/imagenet-vgg-s.mat)
  - [imagenet-vgg-m-2048](models/imagenet-vgg-m-2048.mat)
  - [imagenet-vgg-m-1024](models/imagenet-vgg-m-1024.mat)
  - [imagenet-vgg-m-128](models/imagenet-vgg-m-128.mat)
  - **Citation:** `Return of the Devil in the Details: Delving Deep
    into Convolutional Networks', *Ken Chatfield, Karen Simonyan,
    Andrea Vedaldi, and Andrew Zisserman,* BMVC 2014
    ([BibTex and paper](http://www.robots.ox.ac.uk/~vgg/publications/2014/Chatfield14/)).

- Berkeley
  [Caffe reference models](http://caffe.berkeleyvision.org/getting_pretrained_models.html)
  (version downloaded on July 2014):
  - [imagenet-caffe-ref](models/imagenet-caffe-ref.mat)
  - [imagenet-caffe-alex](models/imagenet-caffe-alex.mat)
  - **Citation:** Please see [Caffe homepage](http://caffe.berkeleyvision.org).

This is a summary of the performance of these models on the ILSVRC
2012 validation data:


|               model|top-1 err.|top-5 err.|  images/s|
|--------------------|----------|----------|----------|
|           caffe-ref|      42.4|      19.6|     132.9|
|          caffe-alex|      42.6|      19.6|     131.4|
|               vgg-s|      36.7|      15.5|     120.0|
|               vgg-m|      37.8|      16.1|     127.6|
|               vgg-f|      41.9|      19.3|     147.0|
|     vgg-verydeep-19|      30.5|      11.3|      40.3|
|     vgg-verydeep-16|      30.9|      11.2|      46.2|

Note that these error rates are computed on a single centre-crop and
are therefore higher than what reported in some publications, where
multiple evaluations per image are combined.

**Example usage.** In order to run, say, `imagenet-vgg-s` on a test
image, use:

    % setup MtConvNet in MATLAB
    run matlab/vl_setupnn

    % download a pre-trained CNN from the web
    urlwrite('http://www.vlfeat.org/sandbox-matconvnet/models/imagenet-vgg-f.mat', ...
      'imagenet-vgg-f.mat') ;
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
that implements a CNN with a simple linear structure (a chain of
layers). It is not needed to use the toolbox, but it simplifies common
examples such as the ones discussed here. See also
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

GPU support in MatConvNet builds on top of MATLAB GPU support in the
Parallel Programming Toolbox. This toolbox requires CUDA-compatible
cards, and you will need a copy of the corresponding
[CUDA devkit](https://developer.nvidia.com/cuda-toolkit-archive) to
compile GPU support in MatConvNet (see [compiling](#compiling)).

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

We gratefully acknowledge the support of NVIDIA Corporation with the
donation of the GPUs used to develop this software.

## Changes

- 1.0-beta7 (September 2014) Adds VGG verydeep models.
- 1.0-beta6 (September 2014) Performance improvements.
- 1.0-beta5 (September 2014) Bugfixes, adds more documentation,
  improves ImageNet example.
- 1.0-beta4 (August 2014) Further cleanup.
- 1.0-beta3 (August 2014) Cleanup.
- 1.0-beta2 (July 2014) Adds a set of standard models.
- 1.0-beta1 (June 2014) First public release
