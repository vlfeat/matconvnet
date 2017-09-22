function y = vl_nnnormalizelp(x,dzdy,varargin)
%VL_NNNORMALIZELP  CNN Lp normalization
%   Y = VL_NNNORMALIZELP(X) normalizes in Lp norm each column of
%   features in the array X, along its third dimension:
%
%       Y(i,j,k) = X(i,j,k) / sum_q (X(i,j,q).^p + epsilon)^(1/p)
%
%   [Y,N] = VL_NNNORMALIZELP(X) returns the array N containing the
%   computed norms.
%
%   DZDX = VL_NNNORMALIZELP(X, DZDY) computes the derivative of the
%   function with respect to X projected onto DZDY.
%
%   VL_NNNORMALIZE(___, 'opts', val, ...) takes the following options:
%
%   `exponent`:: 2
%      The exponent of the Lp norm. Warning: currently only even
%      exponents are supported.
%
%   `p`:: same as exponent
%
%   `epsilon`:: 0.01
%      The constant added to the sum of p-powers before taking the
%      1/p square root (see the formula above).
%
%   `dimensions`:: [3]
%      The list of dimensions along wich to operate. By default,
%      normalization is along the third dimension, usually
%      corresponding to feature channels.
%
%   `spatial`:: `false`
%      If `true`, sum along the two spatial dimensions instead of
%      along the feature channels. This is the same as setting
%      `dimensions` to [1,2].
%
%   See also: VL_NNNORMALIZE().

opts.epsilon = 1e-2 ;
opts.p = 2 ;
opts.spatial = false ;
opts = vl_argparse(opts, varargin, 'nonrecursive') ;

if ~opts.spatial
  massp = sum(x.^opts.p,3) + opts.epsilon ;
else
  massp = sum(sum(x.^opts.p,1),2) + opts.epsilon ;
end
mass = massp.^(1/opts.p) ;
y = bsxfun(@rdivide, x, mass) ;

if nargin < 2 || isempty(dzdy)
  return ;
else
  dzdy = bsxfun(@rdivide, dzdy, mass) ;
  if ~opts.spatial
    tmp = sum(dzdy .* x, 3) ;
  else
    tmp = sum(sum(dzdy .* x, 1),2);
  end
  y = dzdy - bsxfun(@times, tmp, bsxfun(@rdivide, x.^(opts.p-1), massp)) ;
end
