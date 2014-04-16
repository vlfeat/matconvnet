function y = gvec(x, dzdy)
% VL_NNVEC  Neural-network vec operator
%   Y = VL_NNVEC(X) vectorizes the image stack X. If X has
%   dimension H x W x D x N, Y is a vecto stack with dimension HWD x N.
%
%   DYDX = VL_NNVEC(X, DZDY) computes derivative of the network output Z
%   with respect to the input X. DZDY has the same dimension as Y
%   and DYDZ the same dimension as X.

% Author: Andrea Vedaldi

[h,w,d,n] = size(x) ;

if nargin < 2
  y = reshape(x, h*w*d, n) ;
else
  y = reshape(dzdy, h,w,d, n) ;
end