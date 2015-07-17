function y = vl_nnlrelu(x,dzdy,varargin)
% VL_NNLRELU  CNN leaky rectified linear unit
%   Y = VL_NNLRELU(X) applies the leaky rectified linear unit to the
%   data X. X can have arbitrary size. Y is equal to X if X is not
%   smaller than zero; otherwise, Y is equal to X multipied by the
%   leak factor.
%
%   DZDX = VL_NNLRELU(X, DZDY) computes the network derivative DZDX
%   with respect to the input X given the derivative DZDY with respect
%   to the output Y. DZDX has the same dimension as X.
%
%   VL_NNRELU(...,'OPT',VALUE,...) takes the following options:
%
%   `Leak`:: 0.01
%      Leak factor. It should be a non-negative number.
%
%   ADVANCED USAGE
%
%   As a further optimization, in the backward computation it is
%   possible to replace X with Y, namely, if Y = VL_NNLRELU(X), then
%   VL_NNLRELU(X,DZDY) gives the same result as VL_NNLRELU(Y,DZDY).
%   This is useful because it means that the buffer X does not need to
%   be remembered in the backward pass.

% Copyright (C) 2015 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

opts.leak = 0.01 ;
opts = vl_argparse(opts, varargin) ;

if nargin <= 1 || isempty(dzdy)
  y = (1 - opts.leak) + opts.leak * single(x > 0) ;
else
  y = dzdy .* ((1 - opts.leak) + opts.leak * single(x > 0)) ;
end
