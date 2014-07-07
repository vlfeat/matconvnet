function Y = vl_nnsoftmax(X,dzdY)
% VL_NNSOFTMAX  CNN softmax
%    Y = VL_NNSOFTMAX(X) computes the softmax operator of the vector
%    stack X. X has dimension D x N, packing N vectors of dimension D.
%
%    DZDX = VL_NNSOFTMAX(X, DZDY) computes the network ouptut Z
%    derivative with respect to the input X given the derivative DZDY
%    with respect to the output Y. DZDX has the same dimension as X.

% Copyright (C) 2014 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

E = exp(bsxfun(@minus, X, max(X,[],3))) ;
L = sum(E,3) ;
Y = bsxfun(@rdivide, E, L) ;

if nargin <= 1, return ; end

% backward
Y = Y .* bsxfun(@minus, dzdY, sum(dzdY .* Y, 3)) ;
