function Y = vl_nnsoftmaxloss(X,c,dzdy)
% VL_NNSOFTMAXLOSS  CNN combined softmax and logistic loss
%    Y = VL_NNSOFTMAX(X, C) applies the softmax operator followed by
%    the logistic loss the data X. X has dimension H x W x D x N,
%    packing N arrays of W x H D-dimensional vectors.
%
%    C contains the class labels, which should be integer in the range
%    1 to D.  C can be an array with either N elements or with H x W x
%    1 x N dimensions. In the fist case, a given class label is
%    applied at all spatial locations; in the second case, different
%    class labels can be specified for different locations.
%
%    D can be thought of as the number of possible classes and the
%    function computes the softmax along the D dimension. Often W=H=1,
%    but this is not a requirement, as the operator is applied
%    convolutionally at all spatial locations.
%
%    DZDX = VL_NNSOFTMAXLOSS(X, C, DZDY) computes the derivative DZDX
%    of the CNN with respect to the input X given the derivative DZDY
%    with respect to the block output Y. DZDX has the same dimension
%    as X.

% Copyright (C) 2014 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

%X = X + 1e-6 ;
sz = [size(X,1) size(X,2) size(X,3) size(X,4)] ;

% index from 0
c = c - 1 ;

if numel(c) == sz(4)
  % one label per image
  c = reshape(c, [1 1 1 sz(4)]) ;
  c = repmat(c, [sz(1) sz(2)]) ;
else
  % one label per spatial location
  sz_ = [size(c,1) size(c,2) size(c,3) size(c,4)] ;
  assert(isequal(sz_, [sz(1) sz(2) 1 sz(4)])) ;
end

% convert to indeces
c_ = 0:numel(c)-1 ;
c_ = 1 + ...
  mod(c_, sz(1)*sz(2)) + ...
  (sz(1)*sz(2)) * c(:)' + ...
  (sz(1)*sz(2)*sz(3)) * floor(c_/(sz(1)*sz(2))) ;

% compute softmaxloss
Xmax = max(X,[],3) ;
ex = exp(bsxfun(@minus, X, Xmax)) ;

n = sz(1)*sz(2) ;
if nargin <= 2
  t = Xmax + log(sum(ex,3)) - reshape(X(c_), [sz(1:2) 1 sz(4)]) ;
  Y = sum(t(:)) / n ;
else
  Y = bsxfun(@rdivide, ex, sum(ex,3)) ;
  Y(c_) = Y(c_) - 1;
  Y = Y * (dzdy / n) ;
end
