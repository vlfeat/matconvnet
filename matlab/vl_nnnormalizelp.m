function y = vl_nnnormalizelp(x,dzdy,varargin)
%VL_NNNORMALIZELP  CNN Lp normalization
%   Y = VL_NNNORMALIZELP(X) normalizes in Lp norm each spatial
%   location in the array X:
%
%       Y(i,j,k) = X(i,j,k) / sum_q (X(i,j,q).^p + epsilon)^(1/p)
%
%
%   DZDX = VLN_NNORMALIZELP(X, DZDY) computes the derivative of the
%   function with respect to X projected onto DZDY.
%
%   Options:
%
%   `p`:: 2
%      The exponent of the Lp norm. Warning: currently only even
%      exponents are supported.
%
%   `epsilon`: 0.01
%      The constant added to the sum of p-powers before taking the
%      1/p square root (see the formula above).

opts.epsilon = 1e-2 ;
opts.p = 2 ;
opts = vl_argparse(opts, varargin) ;

massp = (sum(x.^opts.p,3) + opts.epsilon) ;
mass = massp.^(1/opts.p) ;
y = bsxfun(@rdivide, x, mass) ;

if nargin < 2 || isempty(dzdy)
  return ;
else
  dzdy = bsxfun(@rdivide, dzdy, mass) ;
  y = dzdy - bsxfun(@times, sum(dzdy .* x, 3), bsxfun(@rdivide, x.^(opts.p-1), massp)) ;
end
