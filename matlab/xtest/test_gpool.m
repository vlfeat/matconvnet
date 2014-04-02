%clear all mex ;
addpath mex

pool = [3 3] ;
gpu = true ;
delta = 0.01 ;
switch 2
  case 1
    x=im2single(imread('cameraman.tif')) ;
  case 2
    x=randn(5,7,2,3,'single') ;
    % make sure that all elements in x are different. in this way,
    % we can compute numerical derivatives reliably by adding a delta < .5.
    x(:) = randperm(numel(x))' ;
end

if ~gpu
  y = gpool(x, pool) ;
else
  y = gpool(gpuArray(x), pool) ;
  y = gather(y) ;
  y_ = gpool(x,pool) ;
end
dzdy = randn(size(y),'single') ;
if ~gpu
  dzdx = gpool(x, pool, dzdy) ;
else
  dzdx = gpool(gpuArray(x), pool, gpuArray(dzdy)) ;
  dzdx = gather(dzdx) ;
end

% numerical derivative
dzdx_=zeros(size(dzdx));
for i=1:numel(x)
  dx = zeros(size(x)) ;
  dx(i) = delta ;
  y_=gpool(x+dx,pool) ;
  dzdx_(i) = dzdx_(i) + sum(sum(sum(sum(dzdy .* (y_ - y)/delta)))) ;
end
assert(max(abs(dzdx(:) - dzdx_(:))) < 1e-2) ;

return

figure(1) ; clf ;
subplot(2,2,1) ; imagesc(a(:,:,1)) ; axis equal ;
subplot(2,2,2) ; imagesc(b(:,:,1)) ; axis equal ;
%subplot(2,2,3) ; imagesc(c(:,:,1)) ; axis equal ; title(sprintf('gpu:%f', gpu_time)) ;
%subplot(2,2,4) ; imagesc(c_cpu(:,:,1)) ; axis equal ; title(sprintf('cpu:%f', cpu_time)) ;
colormap gray ;
