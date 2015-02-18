for explictReset = [false]

  % reset the same GPU device
  for t = 1:5
    if explictReset, clear mex ; end
    gpuDevice(1) ;
    if t > 1, disp(a) ; end
    a = gpuArray(single(1)) ;
    b = gpuArray(single(1)) ;
    c = vl_nnconv(a,b,[],'nocudnn','verbose') ;
  end

  % switch GPU devices
  if gpuDeviceCount > 1
    disp('vl_text_gpureset: test switching GPU device') ;
    for t = 1:gpuDeviceCount
      if explictReset, clear mex ; end
      gpuDevice(t) ;
      a = gpuArray(single(1)) ;
      b = gpuArray(single(1)) ;
      c = vl_nnconv(a,b,[],'nocudnn','verbose') ;
    end
  end
end
