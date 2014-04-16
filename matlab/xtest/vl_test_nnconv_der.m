addpath mex

gpu=false ;
% gpu=true ;
numFilters = 7 ;
height = 6 ;
width = 9 ;
filterHeight = 5 ;
filterWidth = 2 ;
depth = 13 ;
numImages = 4 ;
delta = 2.5 ;

rng(2) ;
x=randn(height,width,depth,numImages,'single') ;
f=randn(filterHeight,filterWidth,depth,numFilters,'single') ;
y=gconv(x,f) ;
[dzdy]=randn(size(y),'single') ;

if gpu
  y = gconv(gpuArray(x),gpuArray(f)) ;
  [dzdf,dzdx] = gconv(gpuArray(x), gpuArray(f), gpuArray(dzdy)) ;
  y = gather(y) ;
  dzdf = gather(dzdf) ;
  dzdx = gather(dzdx) ;
  %y__=gconv(x,f) ;
  %[dzdf__,dzdx__] = gconv(x,f,dzdy) ;
else
  [dzdf,dzdx] = gconv(x,f,dzdy) ;
end


% compute the gradient numerically numerically
dzdf_=zeros(size(dzdf));
for i=1:numel(f)
  df = zeros(size(f)) ;
  df(i) = delta ;
  y_=gconv(x,f+df) ;
  dzdf_(i) = dzdf_(i) + sum(sum(sum(sum(dzdy .* (y_ - y)/delta)))) ;
end

dzdx_=zeros(size(dzdx));
for i=1:numel(x)
  dx = zeros(size(x)) ;
  dx(i) = delta ;
  y_=gconv(x+dx,f) ;
  dzdx_(i) = dzdx_(i) + sum(sum(sum(sum(dzdy .* (y_ - y)/delta)))) ;
end

% run ~/src/vlfeat/toolbox/vl_setup.m
figure(1) ; clf ;
subplot(2,2,1) ;
a=reshape(dzdf,size(dzdf,1),size(dzdf,2),[]) ;
b=reshape(dzdf - dzdf_,size(dzdf,1),size(dzdf,2),[]) ;
vl_imarraysc(abs(cat(2, a, b)),'uniform',true) ;
subplot(2,2,2) ;
hist(abs(dzdf(:) - dzdf_(:))) ;
subplot(2,2,3) ;
hist(abs(dzdx(:) - dzdx_(:))) ;

assert(max(abs(dzdf(:) - dzdf_(:))) < 1e-4*max(abs(dzdf(:)))) ;
assert(max(abs(dzdx(:) - dzdx_(:))) < 1e-4*max(abs(dzdx(:)))) ;


