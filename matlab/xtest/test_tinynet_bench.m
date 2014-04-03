n = 2 ;
T = 40 ;

net = load(fullfile(vlg_root, 'data', 'tinynet_caffe.mat')) ;
load(fullfile(vlg_root, 'data', 'tinynet_caffe_data.mat')) ;
dzdy = zeros(1000,1,'single') ;
dzdy(286) = 1 ;
im = repmat(im, [1 1 1 n]) ;
dzdy = repmat(dzdy, [1 n]) ;

reset(gpuDevice) ;
net_ = tinynet_move(net, 'gpu') ;
im_ = gpuArray(im) ;
dzdy_ = gpuArray(dzdy) ;

res = tinynet(net, im, dzdy) ;
res_ = tinynet(net_, im_, dzdy_) ;
wait(gpuDevice) ;

cpu_times = zeros(T, numel(res)) ;
gpu_times = zeros(T, numel(res)) ;

gpu_time = tic ;
for i=1:T
  res_ = tinynet(net_, im_, dzdy_) ;
  gpu_times(i,:) = [res_.time] + [res_.backwardTime] ;
end
wait(gpuDevice) ;
gpu_time = toc(gpu_time) ;

cpu_time = tic ;
for i=1:T
  res = tinynet(net, im, dzdy) ;
  cpu_times(i,:) = [res.time] + [res.backwardTime] ;
end
cpu_time = toc(cpu_time) ;

fprintf('cpu: %g gpu: %g, speedup: %g\n', cpu_time, gpu_time, cpu_time/gpu_time) ;
fprintf('cpu: %g gpu: %g, speedup: %g\n', sum(cpu_times(:)), ...
        sum(gpu_times(:)), ...
        sum(cpu_times(:))/sum(gpu_times(:))) ;
figure(1) ; clf ;
errorbar([mean(cpu_times) ; mean(gpu_times)]', [std(cpu_times) ; std(gpu_times)]'/sqrt(T)) ;
legend('cpu', 'gpu') ;
title('time per stage') ;
grid on ;
