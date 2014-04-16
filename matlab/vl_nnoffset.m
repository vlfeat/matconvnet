function y = gnoffset(x, param, dzdy)

t = sum(x.^2,3) ;

if nargin <= 2
  y = bsxfun(@minus, x, param(1)*t.^param(2)) ;
else
  y = dzdy - bsxfun(@times, 2*param(1)*param(2)*x, sum(dzdy,3) .* (t.^(param(2)-1))) ;
end