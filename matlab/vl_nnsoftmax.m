function Y = gsoftmax(X,dzdY)
% VL_NNSOFTMAX  Neural-network softmax
%  Y = VL_NNSOFTMAX(X) computes the softmax operator of the vector stack
%  X. X has dimension D x N, packing N vectors of dimension D.
%
%  DZDX = VL_NNSOFTMAX(X, DZDY) computes the network ouptut Z derivative
%  with respect to the input X given the derivative DZDY. DZDX has the
%  same dimension as X.

% Author: Andrea Vedaldi

E = exp(bsxfun(@minus, X, max(X,[],1))) ;
L = sum(E) ;
Y = bsxfun(@rdivide, E, L) ;

if nargin <= 1, return ; end

% backward
Y = Y .* bsxfun(@minus, dzdY, sum(dzdY .* Y,1)) ;
