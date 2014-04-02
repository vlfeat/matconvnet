function y = gvec(x, dzdy)

[h,w,d,n] = size(x) ;

if nargin < 2
  y = reshape(x, h*w*d, n) ;
else
  y = reshape(dzdy, h,w,d, n) ;
end