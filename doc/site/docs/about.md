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

# Copyright and acknowledgments

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
