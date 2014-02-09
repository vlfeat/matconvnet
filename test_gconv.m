clear all mex ;
addpath mex ;

switch 2
  case 1
    a=im2single(imread('cameraman.tif')) ;
    b=im2single(fspecial('laplacian')) ;
  case 2
    a=randn(128,128,32,16,'single') ;
    b=randn(12,12,32,64,'single') ;
end

% do it on the GPU
a_= gpuArray(a) ;
b_= gpuArray(b) ;
c_= gconv(a_,b_) ;
c_= gconv(a_,b_) ;
gpu_time = tic ;
c_= gconv(a_,b_) ;
gpu_time = toc(gpu_time) ;
c = gather(c_) ;

% do it on the CPU
c_cpu = gconv(a,b) ;
c_cpu = gconv(a,b) ;
cpu_time = tic ;
c_cpu = gconv(a,b) ;
cpu_time = toc(cpu_time) ;

figure(1) ; clf ;
subplot(2,2,1) ; imagesc(a(:,:,1)) ; axis equal ;
subplot(2,2,2) ; imagesc(b(:,:,1)) ; axis equal ;
subplot(2,2,3) ; imagesc(c(:,:,1)) ; axis equal ; title(sprintf('%f', gpu_time)) ;
subplot(2,2,4) ; imagesc(c_cpu(:,:,1)) ; axis equal ; title(sprintf('%f', cpu_time)) ;
colormap gray ;

disp('seedup') ;
disp(cpu_time/gpu_time) ;
