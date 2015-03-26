function vl_bench_bnorm

x = randn(64,64,32,32) ;
g = randn(32,1) ;
b = randn(32,1) ;

x = gpuArray(x) ;
g = gpuArray(g) ;
b = gpuArray(b) ;

tic
for t=1:10
y = vl_nnbnorm(x,g,b) ;
end
toc

tic
for t=1:10
y_ = vl_nnbnorm_old(x,g,b) ;
end
toc
mean(abs(y(:)-y_(:)))

dzdy = randn(size(y)) ;

tic
for t=1:10
[a,b,c] = vl_nnbnorm(x,g,b,dzdy) ;
end
toc

tic
for t=1:10
[a_,b_,c_] = vl_nnbnorm_old(x,g,b,dzdy) ;
end
toc
mean(abs(y(:)-y_(:)))
mean(abs(a(:)-a_(:)))
mean(abs(b(:)-b_(:)))
mean(abs(c(:)-c_(:)))