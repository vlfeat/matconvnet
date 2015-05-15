# About MatConvNet

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

<a name='changes'></a>
# Changes

-   1.0-beta12 (May 2015). Added `vl_nnconvt` (convolution transpose or
    deconvolition).
-   1.0-beta11 (April 2015) Added batch normalization, spatial
    normalization, sigmoid, p-distance.  Extended the example training
    code to support multiple GPUs. Significantly improved the tuning
    of the ImageNet and CIFAR examples. Added the CIFAR Network in
    Network model.

    This version changes slightly the structure of `simplenn`. In
    particular, the `filters` and `biases` fields in certain layers
    have been replaced by a `weights` cell array containing both
    tensors, simiplifying a significant amount of code. All examples
    and downloadable models have been updated to reflact this
    change. Models using the old structure format still work but are
    deprecated.

    The `cnn_train` training code example has been rewritten to
    support multiple GPUs.  The inteface is nearly the same, but the
    `useGpu` option has been replaced by a `gpus` list of GPUs to use.

-   1.0-beta10 (March 2015) vl_imreadjpeg works under Windows as well.
-   1.0-beta9 (February 2015) CuDNN support. Major rewrite of the C/CUDA core.
-   1.0-beta8 (December 2014) New website. Experimental Windows support.
-   1.0-beta7 (September 2014) Adds VGG verydeep models.
-   1.0-beta6 (September 2014) Performance improvements.
-   1.0-beta5 (September 2014) Bugfixes, adds more documentation,
    improves ImageNet example.
-   1.0-beta4 (August 2014) Further cleanup.
-   1.0-beta3 (August 2014) Cleanup.
-   1.0-beta2 (July 2014) Adds a set of standard models.
-   1.0-beta1 (June 2014) First public release.

# Copyright

This package was originally created by
[Andrea Vedaldi](http://www.robots.ox.ac.uk/~vedaldi) and Karel Lenc
and it is currently develped by a small community of contributors. It
is distributed under the permissive BSD license (see also the file
`COPYING`):

    Copyright (c) 2014-15 The MatConvNet team.
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

# Acknowledgments

The implementation of the computational blocks in this library, and in
particular of the convolution operators, is inspired by
[Caffe](http://caffe.berkeleyvision.org).

We gratefully acknowledge the support of NVIDIA Corporation with the
donation of the GPUs used to develop this software.
