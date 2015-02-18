clear mex ;
gpuDevice(1) ;
a = gpuArray(single(1)) ;
b = gpuArray(single(1)) ;
c = vl_nnconv(a,b,[],'nocudnn','verbose') ;

% reset GPU device
clear mex ;
gpuDevice(1) ;
a
b
a = gpuArray(single(1)) ;
b = gpuArray(single(1)) ;
c = vl_nnconv(a,b,[],'nocudnn','verbose') ;

% switch GPU device
for t = 1:gpuDeviceCount
  clear mex ;
  gpuDevice(t) ;
  a = gpuArray(single(1)) ;
  b = gpuArray(single(1)) ;
  c = vl_nnconv(a,b,[],'nocudnn','verbose') ;
end
