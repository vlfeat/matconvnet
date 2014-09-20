% VL_NNCONV  CNN convolution
%    Y = VL_NNCONV(X, F, B) computes the convolution of the image stack X
%    with the filter bank F and biases B. If B is the empty matrix,
%    then no biases are added. If F is the empty matrix, then
%    the function does not filter the image, but still adds the
%    biases as well as performing downsampling and padding as explained
%    below.
%
%    [DXDY, DXDF, DXDB] = VL_NNCONV(X, F, B, DZDY) computes the
%    derivatives of the nework output Z w.r.t. the data X and
%    parameters F, B given the derivative w.r.t the output Y. If B is
%    the empty matrix, then DXDB is also empty.
%
%    X is a SINGLE array of dimension H x W x D x N where (H,W) are
%    the height and width of the map stack, D is the image depth
%    (number of feature channels) and N the number of of images in the
%    stack.
%
%    F is a SINGLE array fo dimension FW x FH x D x K where (FH,FW) are
%    the filter height and width and K the number o filters in the
%    bank.
%
%    VL_NNCONV() implements a special `fully-connected' mode: when the
%    support of the filters matches exactly the support of the input
%    image, the code uses an optimized path for faster computation.
%
%    VL_NNCONV(..., 'option', value, ...) takes the following options:
%
%    Stride:: [1]
%      The output stride (downsampling factor). Passing [STRIDEY
%      STRIDEX] allows specifying different subsampling factors for
%      the vertical and horizontal directions.
%
%    Pad:: [0]
%      The amount of input padding. Input images are padded with zeros
%      by this number of pixels before the convolution is
%      computed. Passing [TOP BOTTOM LEFT RIGHT] allows specifying
%      different padding amounts for the top, bottom, left, and right
%      sides respectively.
%
%    The filter size must be not larger than the padded image, i.e.
%
%      1 <= FH <= H + 2*(PADTOP+PADBOTTOM),
%      1 <= FW <= W + 2*(PADLEFT+PADRIGHT).
%
%    The output a is a SINGLE array of dimension YH x YW x K x N of
%    N images with K challens and size:
%
%      YH = floor((H + (PADTOP+PADBOTTOM) - FH)/STRIDEY) + 1,
%      YW = floor((W + (PADLEFT+PADRIGHT) - FW)/STRIDEX) + 1.
%
%    The derivative DZDY has the same dimension of the output Y,
%    the derivative DZDX has the same dimension as the input X, and
%    the derivative DZDF has the the same dimenson as F.

% Copyright (C) 2014 Andrea Vedaldi and Max Jaderberg.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).
