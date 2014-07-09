function Y = vl_nnsoftmaxloss(X,c,dzdy)
% VL_NNSOFTMAXLOSS  CNN combined softmax and logistic loss
%  Y = VL_NNSOFTMAXLOSS(X, C) computes the softmax operator of the
%  vector stack X. X has dimension D x N, packing N vectors of
%  dimension D.
%
%  DZDX = VL_NNSOFTMAX(X, C, DZDY) computes the network ouptut Z
%  derivative with respect to the input X given the derivative DZDY
%  relative to the output Y. DZDX has the same dimension as X.

% Copyright (C) 2014 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

%X = X + 1e-6 ;
sz = [size(X,1) size(X,2) size(X,3) size(X,4)] ;
c_ = c+(0:sz(3):sz(3)*sz(4)-1) ;

Xmax = max(X,[],3) ;
ex = exp(bsxfun(@minus, X, Xmax)) ;

if nargin <= 2
  t = Xmax + log(sum(ex,3)) - reshape(X(:,:,c_), [sz(1:2) 1 sz(4)]) ;
  Y = sum(t(:)) ;
else
  Y = bsxfun(@rdivide, ex, sum(ex,3)) ;
  Y(:,:,c_) = Y(:,:,c_) - 1;
  Y = Y * dzdy ;
end
