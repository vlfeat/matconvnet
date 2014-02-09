clear all mex ;
addpath mex ;
%a=gpuArray(rand(13,11,3,'single'));
%b=gpuArray(ones(3,3,3,'single')) ;
%c=fast_conv(a,b) ;

a=im2single(imread('cameraman.tif')) ;
b=im2single(fspecial('laplacian')) ;
%b=ones(10,'single') ;

a_= gpuArray(a) ;
b_= gpuArray(b) ;
c_= fast_conv(a_,b_) ;
c = gather(c_) ;

figure(1) ; clf ;
subplot(1,3,1) ; imagesc(a) ; axis equal ;
subplot(1,3,2) ; imagesc(b) ; axis equal ;
subplot(1,3,3) ; imagesc(c) ; axis equal ;

colormap gray ;
