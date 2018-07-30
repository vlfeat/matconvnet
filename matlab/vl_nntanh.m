function out = vl_nntanh(x,dzdy)
%VL_NNTANH CNN tanh nonlinear unit.
%   Y = VL_NNTANH(X) computes the hyperbolic tanget of the data X. X can
%   have an arbitrary size. The tanh is defined as follows:
%
%     TANH(X) = (EXP(X) - EXP(-X)) / (EXP(X) + EXP(-X)).
%
%   DZDX = VL_NNTANH(X, DZDY) computes the derivative of the
%   block projected onto DZDY. DZDX and DZDY have the same
%   dimensions as X and Y respectively.

% Copyright (C) 2016 Hemanth Venkateswara.
% All rights reserved.
%

% y = (1 - exp(-2*x)) ./ (1 + exp(-2*x)); % This was blowing up
% if any(isnan(y(:)))
%     error('nan has occured');
% end
y = tanh(x);

if nargin <= 1 || isempty(dzdy)
  out = y ;
else
  out = dzdy .* (1 - y.^2) ;
end
