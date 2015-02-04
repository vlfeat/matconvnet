function vl_testder(g,x,dzdy,dzdx,delta,tau)

if nargin < 5
  delta = 1e-3 ;
end

if nargin < 6
  tau = [] ;
end

if ~iscell(x)
  x = {x};
  dzdx = {dzdx};
  gc = @(x) g(x{1});
else
  gc = g;
end

dzdy = gather(dzdy) ;
dzdx = cellfun(@gather, dzdx, 'UniformOutput', false) ;

y = gather(gc(x)) ;
for ii=1:numel(x)
  dzdx_=zeros(size(dzdx{ii}));
  for i=1:numel(x{ii})
    x_ = x ;
    x_{ii}(i) = x_{ii}(i) + delta ;
    y_ = gather(gc(x_)) ;
    factors = dzdy .* (y_ - y)/delta ;
    dzdx_(i) = dzdx_(i) + sum(factors(:)) ;
  end
  vl_testsim(dzdx{ii}, dzdx_, tau);
end

