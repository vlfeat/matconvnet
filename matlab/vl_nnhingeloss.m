function Y = vl_nnhingeloss(X,c,dzdy)
% VL_NNHINGELOSS  Hinge loss
%    Y = VL_NNHINGELOSS(X, C) applies the the hinge loss to the data
%    X. X has dimension H x W x D x N, packing N arrays of W x H
%    D-dimensional vectors.
%
%    C contains the class labels, which should be integers in the range
%    1 to D. C can be an array with either N elements or with H x W x
%    1 x N dimensions. In the fist case, a given class label is
%    applied at all spatial locations; in the second case, different
%    class labels can be specified for different locations.
%
%    D can be thought of as the number of possible classes and the
%    function computes the softmax along the D dimension. Often W=H=1,
%    but this is not a requirement, as the operator is applied
%    convolutionally at all spatial locations.
%
%    DZDX = VL_NNHINGELOSS(X, C, DZDY) computes the derivative DZDX of the
%    CNN with respect to the input X given the derivative DZDY with
%    respect to the block output Y. DZDX has the same dimension as X.

% Copyright (C) 2015 James Thewlis & Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

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

% convert to indices
% such that X(c_) returns only elements with the correct label
c_ = 0:numel(c)-1 ;
c_ = 1 + ...
  mod(c_, sz(1)*sz(2)) + ...
  (sz(1)*sz(2)) * c(:)' + ...
  (sz(1)*sz(2)*sz(3)) * floor(c_/(sz(1)*sz(2))) ;

% multiply by -1 for inputs where the label is incorrect
X_ = -X ;
X_(c_) = X_(c_) * -1 ;

n = sz(1)*sz(2) ;
if nargin <= 2
  % Effectively does (for the non-spatial case):
  %   y(1,1,j,i) = max(0, 1 - X(1,1,j,i) * ((class(i) == j) * 2 - 1))^2
  y = max(0, 1 - X_) .^ 2 ;
  Y = sum(y(:)) / n ;
else
  % Computes
  %   y(1,1,j,i) = - 2 * t * max(0, 1 - X(1,1,j,i) * t)
  % Where t = (class(i) == j) * 2 - 1
  y = 2 * max(0, 1 - X_) ;
  y(c_) = -1 * y(c_) ;
  Y = y * dzdy / n;
end
