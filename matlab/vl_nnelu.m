function y = vl_nnelu(x,dzdy,varargin)
%VL_NNELU CNN rectified linear unit.
%   Y = VL_NNELU(X) applies the rectified linear unit to the data
%   X. X can have arbitrary size.
%
%   DZDX = VL_NNELU(X, DZDY) computes the derivative of the block
%   projected onto DZDY. DZDX and DZDY have the same dimensions as
%   X and Y respectively.
%
%   VL_NNELU(...,'OPT',VALUE,...) takes the following options:
%
%   `Leak`:: 0
%      Set the leak factor, a non-negative number. Y is equal to X if
%      X is not smaller than zero; otherwise, Y is equal to X
%      multipied by the leak factor. By default, the leak factor is
%      zero; for values greater than that one obtains the leaky ELU
%      unit.
%
%   ADVANCED USAGE
%
%   As a further optimization, in the backward computation it is
%   possible to replace X with Y, namely, if Y = VL_NNELU(X), then
%   VL_NNELU(X,DZDY) gives the same result as VL_NNELU(Y,DZDY).
%   This is useful because it means that the buffer X does not need to
%   be remembered in the backward pass.

% Copyright (C) 2014-15 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

opts.alpha = 1;
opts = vl_argparse(opts, varargin) ;

fx = x .* (x > 0) + opts.alpha * (exp(x) - 1) .* (x <= 0);
if nargin <= 1 || isempty(dzdy)
    y = fx;
else
    y = dzdy .* ((x > 0) + (x <= 0) .* (fx + opts.alpha));
end
