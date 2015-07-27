%VL_NNCONV CNN convolution.
%   Y = VL_NNCONV(X, F, B) computes the convolution of the image stack X
%   with the filter bank F and biases B. If B is the empty matrix,
%   then no biases are added. If F is the empty matrix, then
%   the function does not filter the image, but still adds the
%   biases as well as performing downsampling and padding as explained
%   below.
%
%   X is a SINGLE array of dimension H x W x D x N where (H,W) are
%   the height and width of the image stack, D is the image depth
%   (number of feature channels) and N the number of images in the
%   stack.
%
%   F is a SINGLE array of dimension FW x FH x FD x K where (FH,FW)
%   are the filter height and width and K the number o filters in the
%   bank. D is the depth of each filter and must match the depth D of
%   X. Alternatively, FD can *divide* the depth D; in this case,
%   filters are assumed to form G=D/FD *groups* of equal size (where
%   G must divide K). Each group of filters works on a consecutive
%   subset of feature channels of the input array X.
%
%   [DZDX, DZDF, DZDB] = VL_NNCONV(X, F, B, DZDY) computes the
%   derivatives of the block projected onto DZDY. DZDX, DZDF, and
%   DZDB, and DZDY have the same dimensions as X, F, B, and Y
%   repsectively. In particular, if B is the empty matrix, then DZDB
%   is also empty.
%
%   VL_NNCONV() implements a special `fully-connected' mode: when the
%   support of the filters matches exactly the support of the input
%   image, the code uses an optimized path for faster computation.
%
%   VL_NNCONV(..., 'option', value, ...) takes the following options:
%
%   `Stride`:: 1
%     The output stride or downsampling factor. If the value is a
%     scalar, then the same stride is applied to both vertical and
%     horizontal directions; otherwise, passing [STRIDEY STRIDEX]
%     allows specifying different downsampling factors for each
%     direction.
%
%   `Pad`:: 0
%     The amount of input padding. Input images are padded with zeros
%     by this number of pixels before the convolution is
%     computed. Passing [TOP BOTTOM LEFT RIGHT] allows specifying
%     different padding amounts for the top, bottom, left, and right
%     sides respectively. Passing a single scalar applies the same
%     padding to all borders.
%
%   The filter size must be not larger than the padded image, i.e.
%
%     1 <= FH <= H + 2*(PADTOP+PADBOTTOM),
%     1 <= FW <= W + 2*(PADLEFT+PADRIGHT).
%
%   The output a is a SINGLE array of dimension YH x YW x K x N of
%   N images with K challens and size:
%
%     YH = floor((H + (PADTOP+PADBOTTOM) - FH)/STRIDEY) + 1,
%     YW = floor((W + (PADLEFT+PADRIGHT) - FW)/STRIDEX) + 1.
%
%   ## CUDNN SUPPORT
%
%   If compiled in, the function will use cuDNN convolution routines
%   (with the exception of asymmetric left-right or top-bottom
%   padding and a few corner cases such as 1x1 filters in Linux that
%   trigger current bugs in cuDNN). You can use the 'NoCuDNN' option
%   to disable cuDNN or 'cuDNN' to activate it back again (the choice
%   sticks until MATLAB purges the MEX files for any reason).

% Copyright (C) 2014 Andrea Vedaldi and Max Jaderberg.
% Copyright (C) 2015 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).
