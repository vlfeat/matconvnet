switch 1
  case 1
    a=im2single(imread('cameraman.tif')) ;
    b=im2single(fspecial('laplacian')) ;
    b=im2single(fspecial('sobel')) ;
  case 2
    a=randn(128,128,32,16,'single') ;
    b=randn(12,12,32,64,'single') ;
end

% do it on the CPU
c_cpu = gconv(a,b) ;
cpu_time = tic ;
c_cpu = gconv(a,b) ;
for t=1:5; c=gconv(a,b) ; end
cpu_time = toc(cpu_time) ;

% do it on the GPU
a_= gpuArray(a) ;
b_= gpuArray(b) ;
c_=gconv(a_,b_) ;
gpu_time = tic ;
for t=1:5; c_= gconv(a_,b_) ; end
gpu_time = toc(gpu_time) ;
c = gather(c_) ;

figure(1) ; clf ;
subplot(2,2,1) ; imagesc(a(:,:,1)) ; axis equal ;
subplot(2,2,2) ; imagesc(b(:,:,1)) ; axis equal ;
subplot(2,2,3) ; imagesc(c(:,:,1)) ; axis equal ; %title(sprintf('gpu:%f', gpu_time)) ;
subplot(2,2,4) ; imagesc(c_cpu(:,:,1)) ; axis equal ;% title(sprintf('cpu:%f', cpu_time)) ;
colormap gray ;

if 0
fprintf('difference: %g\n', max(abs(c(:)-c_cpu(:)))) ;
fprintf('speedup: %f\n',cpu_time/gpu_time) ;
end
