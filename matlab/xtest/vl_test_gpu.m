speedup = [] ;
psize = [] ;
T = 20 ;

for n=2.^(8:10)
  m = 1 ;
  m = n ;
  A = randn(n,'single');
  B = randn(n,m,'single');
  A_ = gpuArray(A) ;
  B_ = gpuArray(B) ;

  C = A*B ;
  C_ = A_*B_ ;
  wait(gpuDevice) ;

  cpu_time = tic ;
  for i=1:T
    C = A*B ;
    B = A*C ;
  end
  cpu_time = toc(cpu_time) ;

  gpu_time = tic ;
  for i=1:T
    C_ = A_*B_ ;
    B_ = A_*C_ ;
  end
  wait(gpuDevice) ;
  gpu_time = toc(gpu_time) ;

  fprintf('cpu: %g gpu: %g, speedup: %g\n', cpu_time, gpu_time, cpu_time/gpu_time) ;

  speedup(end+1) = cpu_time/gpu_time ;
  psize(end+1) = n ;

  figure(1) ; clf ; plot(psize, speedup) ; grid on ; drawnow ;
end
