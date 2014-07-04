function vl_testsim(a,b,tau)
a = gather(a) ;
b = gather(b) ;
assert(isequal(size(a),size(b))) ;
delta = a - b ;
%max(abs(a(:)-b(:)))
if nargin < 3
  maxv = max([max(a(:)), max(b(:))]) ;
  minv = min([min(a(:)), min(b(:))]) ;
  tau = 1e-2 * (maxv - minv) + 1e-4 * max(maxv, -minv) ;
end
assert(all(abs(a(:)-b(:)) < tau)) ;
