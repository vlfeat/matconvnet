%VL_NNBILIEARSAMPLER Differentiable bilinear interpolation.
%   Y = VL_NNBILINEARSAMPLER(X,GRID) does bilinear interpolation on X,
%   according to the texture map (as defined in the sampling grid
%   GRID).
%
%   X is a array of dimension Hi x Wi x D x Ni, where (H,W) are the
%   height and width of the feature-map, D is the number of channels,
%   and Ni, the number of images in the stack.
%
%   GRID is an array of dimension Ho x Wo x 2 x Ng, where (Ho,Wo) are
%   the height and width of the output feature-map, 2 is for the two
%   spatial dimensions corresponding to Ho and Wo respectively, and Ng
%   is the number of transforms.  (Ho,Wo) need to be equal to (Hi,Wi).
%   Further, Ng can be a multiple of Ni: in this case, it is assumed
%   that there are Ng/Ni transforms per input image, hence, the
%   transforms [1 ... Ng/Ni] are applied to the first image, [Ng/Ni+1
%   ... 2*Ng/Ni] are applied to the second image, etc.
%
%   Y is an array of size Ho x Wo x D x Ng. Note, the SAME transform
%   is applied to all the D channels of a given image.
%
%   [DZDX, DZDGRID] = VL_NNBILINEARSAMPLER(X, GRID, DZDY) computes the
%   derivatives of the block projected onto DZDY.  DZDX,DZDGRID,DZDY
%   have the same dimensions as X,GRID and Y respectively.
%
%   ## CUDNN SUPPORT
%   If compiled in, the function will use cuDNN functions for
%   bilinear interpolation.
%   Note: there are some known issues when using more than 4 channels,
%         with cuDNN.
%   You can use the 'NoCudnn' option to disable cuDNN or 'Cudnn'
%   to activate it back again
%   (the choice sticks until MATLAB purges the MEX files for any reason).

% Copyright (C) 2016 Ankush Gupta, Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).
