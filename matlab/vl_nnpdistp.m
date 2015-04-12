function y = vl_nnpdistp(x, x0, p, dzdy)
% VL_NNPDISTP  CNN p-power p-distance from target
%    VL_NNPDISTP(X, X0, P) computes the P distance raised to the
%    P-power of each feature vector in X to the corresponding feature
%    vector in X0:
%
%      Y(i,j,1) = SUM_d (X(i,j,d) - X0(i,j,d))^P
%
%    X0 should have the same size as X and Y has the same height and
%    width as X, but depth equal to 1.
%
%    DZDX = VL_NNPDISTP(X, X0, P, DZDY) computes the derivative.

if nargin <= 3 || isempty(dzdy)
  if p == 1
    y = sum(abs(x - x0),3) ;
  elseif p == 2
    d = x - x0 ;
    y = sum(d.*d,3) ;
  else
    % general version
    y = sum(abs(x - x0).^p,3) ;
  end
else
  if p == 1
    y = bsxfun(@times, dzdy, sign(x - x0)) ;
  elseif p == 2
    y = bsxfun(@times, 2 * dzdy, x - x0) ;
  else
    y = bsxfun(@times, p * dzdy, abs(x - x0).^(p-2) .* (x - x0)) ;
  end
end
