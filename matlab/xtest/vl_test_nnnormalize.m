depth = 3 ;
kappa = .1 ;
alpha = .25 ;
beta = .5 ;
param = [depth, kappa, alpha, beta] ;
x = randn(2,3,10,2,'single') ;
y = gnormalize(x, param) ;
dzdy = randn(size(y),'single') ;
dzdx = gnormalize(x, param, dzdy) ;

y_ = 0*y ;

for i=1:size(x,1)
  for j=1:size(x,2)
    for n=1:size(x,4)
      t = zeros(1,1,size(x,3),1) ;
      t(1,1,:,1) = (kappa + alpha*conv(squeeze(x(i,j,:,n)).^2, ones(depth,1), 'same')).^(-beta) ;
      y_(i,j,:,n) = x(i,j,:,n) .* t ;
    end
  end
end

y__ = gnormalize(gpuArray(x),param,'verbose') ;
dzdx__ = gnormalize(gpuArray(x), param, gpuArray(dzdy)) ;

if 0
  x = x(:) ;
  y = y(:) ;
  y_ = y_(:) ;
  [y y_]
end