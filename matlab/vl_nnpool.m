% VL_NNPOOL  CNN pooling
%    Y = VL_NNPOOL(X, POOL) applies the pooling operator to all
%    channels of the data X using a square filter of size POOL. X is a
%    SINGLE array of dimension H x W x D x N where (H,W) are the
%    height and width of the map stack, D is the image depth (number
%    of feature channels) and N the number of of images in the stack.
%
%    Y = VL_NNPOOL(X, [POOLY, POOLX]) uses a rectangular filter of
%    height POOLY and width POOLX.
%
%    DZDX = VL_NNPOOL(X, POOL, DZDY) computes the derivatives of
%    the network output Z w.r.t. the data X given the derivative DZDY
%    w.r.t the max-pooling output Y.
%
%    VL_NNCONV(..., 'option', value, ...) takes the following options:
%
%    Stride:: [1]
%      The output stride (downsampling factor). It can be either a
%      scalar for isotropic downsampling or a vector [STRIDEY
%      STRIDEX].
%
%    Pad:: [0]
%      The amount of input padding. Input images are padded with zeros
%      by this number of pixels on all sides before the convolution is
%      computed. It can also be a vector [TOP BOTTOM LEFT RIGHT] to
%      specify a different amount of padding in each direction. The
%      size of the poolin filter has to exceed the padding.
%
%    Method:: ['max']
%      Specify method of pooling. It can be either 'max' (retain max value
%      over the pooling region per channel) or 'avg' (compute the average
%      value over the poolling region per channel).
%
%    The pooling window must be not larger than the padded image, i.e.
%
%      1 <= POOLY <= HEIGHT + (PADTOP + PADBOTTOM),
%      1 <= POOLX <= WIDTH + (PADLEFT + PADRIGHT).
%
%    The output a is a SINGLE array of dimension YH x YW x K x N of N
%    images with K channels and size:
%
%      YH = floor((H + (PADTOP+PADBOTTOM) - POOLY)/STRIDEY) + 1,
%      YW = floor((W + (PADLEFT+PADRIGHT) - POOLX)/STRIDEX) + 1.
%
%    The derivative DZDY has the same dimension of the output Y and
%    the derivative DZDX has the same dimension as the input X.

% Copyright (C) 2014 Andrea Vedaldi, Karel Lenc, and Max Jaderberg.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

