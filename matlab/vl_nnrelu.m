function y = vl_nnrelu(x,dzdy)
% VL_NNRELU  Neural-network rectified linear unit
%  Y = VL_NNRELU(X) applies the rectified linear unit to the data
%  X. X can have arbitrary size.
%
%  DZDX = VL_NNRELU(X, DZDY) computes the network ouptut Z derivative
%  with respect to the input X given the derivative DZDY. DZDX has the
%  same dimension as X.

if nargin <= 1
  y = max(x,0) ;
else
  y = dzdy ;
  y(x <= 0) = 0 ;
end
