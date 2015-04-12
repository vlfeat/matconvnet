function y = vl_nnpdist(x, x0, p, dzdy)
% VL_NNPDIST  CNN p-distance from target
%    VL_NNPDIST(...) is the same as VL_NNPDIST(...) with the
%    difference that the norm (not raised to the P-power) is returned:
%
%      Y(i,j,1) = (SUM_d (X(i,j,d) - X0(i,j,d))^P)^1/P

if nargin <= 3 || isempty(dzdy)
  if p == 1
    y = vl_nnpdistp(x, x0, 1) ;
  elseif p == 2
    d = x - x0 ;
    y = sqrt(sum(d.*d,3)) ;
  else
    y = sum(abs(x - x0).^p,3).^(1/p) ;
  end
else
  if p == 1
    y = vl_nnpdistp(x, x0, 1, dzdy) ;
  elseif p == 2
    d = x - x0 ;
    y = sum(d.*d,3).^(-0.5) ;
    y = bsxfun(@times, dzdy .* y,  d) ;
  else
    y = sum(abs(x - x0).^p,3).^((1-p)/p) ;
    y = bsxfun(@times, dzdy .* y, abs(x - x0).^(p-2) .* (x - x0)) ;
  end
end
