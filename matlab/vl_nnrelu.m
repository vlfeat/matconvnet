function y = vl_nnrelu(x,dzdy)
% VL_NNRELU  CNN rectified linear unit
%   Y = VL_NNRELU(X) applies the rectified linear unit to the data
%   X. X can have arbitrary size.
%
%   DZDX = VL_NNRELU(X, DZDY) computes the network derivative DZDX
%   with respect to the input X given the derivative DZDY with respect
%   to the output Y. DZDX has the same dimension as X.
%
%   ADVANCED USAGE
%
%   As a further optimization, in the backward computation it is
%   possible to replace X with Y, namely, if Y = VL_NNRELU(X), then
%   VL_NNRELU(X,DZDY) gives the same result as VL_NNRELU(Y,DZDY).
%   This is useful because it means that the buffer X does not need to
%   be remembered in the backward pass.

% Copyright (C) 2014 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

if nargin <= 1 || isempty(dzdy)
  y = max(x, single(0)) ;
else
  y = dzdy .* (x > single(0)) ;
end
