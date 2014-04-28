n = 3 ;
T = 40 ;

net = load(fullfile(vl_rootnn, 'data', 'tinynet_caffe.mat')) ;
load(fullfile(vl_rootnn, 'data', 'tinynet_caffe_data.mat')) ;
dzdy = zeros(1,1,1000,'single') ;
dzdy(286) = 1 ;
im = repmat(im, [1 1 1 n]) ;
dzdy = repmat(dzdy, [1 n]) ;

reset(gpuDevice) ;
net_ = vl_simplenn_move(net, 'gpu') ;
im_ = gpuArray(im) ;
dzdy_ = gpuArray(dzdy) ;

res_ = vl_simplenn(net_, im_) ;
res = vl_simplenn(net, im) ;
wait(gpuDevice) ;

cpu_times = zeros(T, numel(res)) ;
gpu_times = zeros(T, numel(res)) ;

gpu_time = tic ;
for i=1:T
  %res_ = tinynet(net_, im_, dzdy_) ;
  res_ = vl_simplenn(net_, im_) ;
  gpu_times(i,:) = [res_.time] + [res_.backwardTime] ;
end
wait(gpuDevice) ;
gpu_time = toc(gpu_time) ;

cpu_time = tic ;
for i=1:T
  %res = tinynet(net, im, dzdy) ;
  res = vl_simplenn(net, im) ;
  cpu_times(i,:) = [res.time] + [res.backwardTime] ;
end
cpu_time = toc(cpu_time) ;

fprintf('cpu: %g gpu: %g, speedup: %g\n', cpu_time, gpu_time, cpu_time/gpu_time) ;
fprintf('cpu: %g gpu: %g, speedup: %g\n', sum(sum(cpu_times)), ...
        sum(sum(gpu_times)), ...
        sum(sum(cpu_times))/sum(sum(gpu_times))) ;
figure(1) ; clf ;
errorbar([mean(cpu_times) ; mean(gpu_times)]', [std(cpu_times) ; std(gpu_times)]'/sqrt(T)) ;
legend('cpu', 'gpu') ;
title('time per stage') ;
grid on ;
