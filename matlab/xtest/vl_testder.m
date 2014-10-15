function vl_testder(g,x,dzdy,dzdx,delta,tau)

if nargin < 5
  delta = 1e-3 ;
end

if nargin < 6
  tau = [] ;
end

dzdy = gather(dzdy) ;
dzdx = gather(dzdx) ;

y = gather(g(x)) ;
dzdx_=zeros(size(dzdx));
for i=1:numel(x)
  x_ = x ;
  x_(i) = x_(i) + delta ;
  y_ = gather(g(x_)) ;
  factors = dzdy .* (y_ - y)/delta ;
  dzdx_(i) = dzdx_(i) + sum(factors(:)) ;
end
vl_testsim(dzdx, dzdx_, tau);

