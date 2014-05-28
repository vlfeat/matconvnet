function vl_testsim(a,b,tau)
a = gather(a) ;
b = gather(b) ;
assert(isequal(size(a),size(b))) ;
delta = a - b ;
%max(abs(a(:)-b(:)))
if nargin < 3
  tau = 1e-2 * (max([a(:) ; b(:)]) - min([a(:) ; b(:)])) ;
end
assert(all(abs(a(:)-b(:)) < tau)) ;
