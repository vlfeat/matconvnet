addpath mex

gpu=false ;
gpu=true ;

rng(0) ;
x=randn(5,5,1,1,'single') ;
f=ones(3,3,1,1,'single') ;
y=gconv(x,f) ;
[dzdy]=randn(size(y),'single') ;

if gpu
  [dzdf,dzdx] = gconv(gpuArray(x), gpuArray(f), gpuArray(dzdy)) ;
  dzdf = gather(dzdf) ;
  dzdx = gather(dzdx) ;
else
  [dzdf,dzdx] = gconv(x,f,dzdy) ;
end
  %[dzdf] = gconv(x,f,dzdy) ;

% compute it numerically
dzdf_=zeros(size(dzdf));
delta = .01;
for i=1:numel(f)
  df = zeros(size(f)) ;
  df(i) = delta ;
  y_=gconv(x,f+df) ;
  dzdf_(i) = dzdf_(i) + sum(sum(dzdy .* (y_ - y)/delta)) ;
end

disp(dzdf)
disp(dzdf_)

dzdx_=zeros(size(dzdx));
delta = .01;
for i=1:numel(x)
  dx = zeros(size(x)) ;
  dx(i) = delta ;
  y_=gconv(x+dx,f) ;
  dzdx_(i) = dzdx_(i) + sum(sum(dzdy .* (y_ - y)/delta)) ;
end

disp(dzdf)
disp(dzdf_)

disp(dzdx)
disp(dzdx_)



