function [y,dzdw,dzdb] = vl_nnfull(x, w, b, dzdy)
% VL_NNFULL   Neural-network fully-connected linear layer
%  Y = VL_NNFULL(X, W, B) computes the fully connected layer for the
%  vector stack X. X is a D x N matrix stacking N
%  input vectors of dimension D. W is a P x D matrix and B a P x 1  vector of linear
%  weights and biases. The output Y contains P x N processed vector.
%
%  [DZDX, DZDW, DZDB] = VL_NNFULL(X, W, B, DZDY) computes the derivative of
%  the network ouptut Z with respect to the input X given the derivative DZDY.
%  DZDX has the same dimension as X.
%
%  Data should be of class SINGLE and can be a gpuArray.

% Author: Andrea Vedaldi

y = bsxfun(@plus, w * x, b) ;

if nargin <= 3, return ; end

% backward
y = (dzdy' * w)' ;
dzdw = dzdy * x' ;
dzdb = sum(dzdy,2) ;
