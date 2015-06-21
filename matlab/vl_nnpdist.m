function y = vl_nnpdist(x, x0, p, varargin)
% VL_NNPDIST  CNN p-distance from target
%    VL_NNPDIST(X, X0, P) computes the P distance raised of each feature
%    vector in X to the corresponding feature vector in X0:
%
%      Y(i,j,1) = (SUM_d (X(i,j,d) - X0(i,j,d))^P)^1/P
%
%    X0 should have the same size as X; the outoput Y has the same
%    height and width as X, but depth equal to 1. Optionally, X0 can
%    be a 1 x 1 x D x N array, in which case the same target feature
%    vector in X0 is compared to all feature vectors in X.
%
%    Setting the `noRoot` option to `true` does not take the 1/P power
%    in the formula, computing instead
%
%      Y(i,j,1) = SUM_d (X(i,j,d) - X0(i,j,d))^P
%
%    For example, `vn_nnpdist(x, x0, 2, 'noRoot', true)` computes the
%    squared L2 distance.
%
%    Options:
%
%    `NoRoot`:: `false`
%       If set to true, compute the P-distance to the P-th power.
%
%    `Epsilon`:: 1e-6
%       When computing derivatives, quantities that are divided in are
%       lower boudned by this value. For example, the L2 distance is
%       not smooth at the origin; this option prevents the
%       derivative from diverging.
%
%
%    DZDX = VL_NNPDISTP(X, X0, P, DZDY) computes the derivative.


% -------------------------------------------------------------------------
%                                                             Parse options
% -------------------------------------------------------------------------

opts.noRoot = false ;
opts.epsilon = 1e-6 ;
backMode = numel(varargin) > 0 && ~ischar(varargin{1}) ;
if backMode
  dzdy = varargin{1} ;
  opts = vl_argparse(opts, varargin(2:end)) ;
else
  dzdy = [] ;
  opts = vl_argparse(opts, varargin) ;
end

% -------------------------------------------------------------------------
%                                                             Parse options
% -------------------------------------------------------------------------

d = bsxfun(@minus, x, x0) ;

if ~opts.noRoot
  if isempty(dzdy)
    if p == 1
      y = sum(abs(d),3) ;
    elseif p == 2
      y = sqrt(sum(d.*d,3)) ;
    else
      y = sum(abs(d).^p,3).^(1/p) ;
    end
  else
    if p == 1
      y = bsxfun(@times, dzdy, sign(d)) ;
    elseif p == 2
      y = max(sum(d.*d,3), opts.epsilon).^(-0.5) ;
      y = bsxfun(@times, dzdy .* y,  d) ;
    elseif p < 1
      y = sum(abs(d).^p,3).^((1-p)/p) ;
      y = bsxfun(@times, dzdy .* y, max(abs(d), opts.epsilon).^(p-1) .* sign(d)) ;
    else
      y = max(sum(abs(d).^p,3), opts.epsilon).^((1-p)/p) ;
      y = bsxfun(@times, dzdy .* y, abs(d).^(p-1) .* sign(d)) ;
    end
  end
else
  if isempty(dzdy)
    if p == 1
      y = sum(abs(d),3) ;
    elseif p == 2
      y = sum(d.*d,3) ;
    else
      y = sum(abs(d).^p,3) ;
    end
  else
    if p == 1
      y = bsxfun(@times, dzdy, sign(d)) ;
    elseif p == 2
      y = bsxfun(@times, 2 * dzdy, d) ;
    elseif p < 1
      y = bsxfun(@times, p * dzdy, max(abs(d), opts.epsilon).^(p-1) .* sign(d)) ;
    else
      y = bsxfun(@times, p * dzdy, abs(d).^(p-1) .* sign(d)) ;
    end
  end
end
