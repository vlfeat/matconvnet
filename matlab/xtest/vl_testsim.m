function testsim(a,b)
a = gather(a) ;
b = gather(b) ;
assert(isequal(size(a),size(b))) ;
delta = a - b ;
%max(abs(a(:)-b(:)))
assert(all(abs(a(:)-b(:)) < 1e-1)) ;
