function y = vl_nnspnorm(x, param, dzdy)
% VL_NNSPNORM  CNN spaital normalization
%    Y = VL_NNSPNORM(X, PARAM) computes the spatial normalization of the
%    data X with parameters PARAM = [PH PW ALPHA BETA]. Here PH and PW
%    define the size of the spatial neighbourhood used for nomalization.
%
%    For each feature channel, the function computes the sum of
%    squares of X inside each rectangle, N2(i,j). It then divides each
%    element of X as follows:
%
%       Y(i,j) = X(i,j) / (1 + ALPHA * N2(i,j))^BETA.
%
%    DZDX = VL_NNSPNORM(X, PARAM, DZDY) computes the derivative DZDX
%    of the block followed by a function whose derivative is DZDY.

% Copyright (C) 2015 Andrea Vedaldi, Karel Lenc, and Max Jaderberg.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

pad = floor((param(1:2)-1)/2) ;
pad = [pad ; param(1:2)-1-pad] ;

n2 = vl_nnpool(x.*x, param(1:2), 'method', 'avg', 'pad', pad) ;
f = 1 + param(3) * n2 ;

if nargin <= 2 || isempty(dzdy)
  y = f.^(-param(4)) .* x ;
else
  t = vl_nnpool(x.*x, param(1:2), f.^(-param(4)-1) .* dzdy .* x, 'method', 'avg', 'pad', pad) ;
  y = f.^(-param(4)) .* dzdy - 2 * param(3)*param(4) * x .* t ;
end