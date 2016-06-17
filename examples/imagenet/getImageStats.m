function [averageImage, rgbMean, rgbCovariance] = getImageStats(images, varargin)
%GETIMAGESTATS  Get image statistics

opts.gpus = [] ;
opts.batchSize = 256 ;
opts.fullImageSize = 256 ;
opts.imageSize = [224 224] ;
opts.numThreads = 6 ;
opts = vl_argparse(opts, varargin) ;

avg = {} ;
rgbm1 = {} ;
rgbm2 = {} ;

numGpus = numel(opts.gpus) ;
if numGpus > 0
  fprintf('%s: resetting GPU device\n', mfilename) ;
  gpuDevice(opts.gpus(1))
end

for t=1:opts.batchSize:numel(images)
  time = tic ;
  batch = t : min(t+opts.batchSize-1, numel(images)) ;
  fprintf('collecting image stats: batch starting with image %d ...', batch(1)) ;

  data = getImageBatch(images(batch), ...
                       'numThreads', opts.numThreads, ...
                       'jitter', false, ...
                       'fullImageSize', opts.fullImageSize, ...
                       'imageSize', opts.imageSize) ;

  if numGpus > 0
    data = gpuArray(data) ;
  end

  z = reshape(permute(data,[3 1 2 4]),3,[]) ;
  n = size(z,2) ;
  avg{end+1} = mean(data, 4) ;
  rgbm1{end+1} = sum(z,2)/n ;
  rgbm2{end+1} = z*z'/n ;
  time = toc(time) ;
  fprintf(' %.1f Hz\n', numel(batch) / time) ;
end

averageImage = gather(mean(cat(4,avg{:}),4)) ;
rgbm1 = mean(cat(2,rgbm1{:}),2) ;
rgbm2 = mean(cat(3,rgbm2{:}),3) ;
rgbMean = gather(rgbm1) ;
rgbCovariance = gather(rgbm2 - rgbm1*rgbm1') ;

if numGpus > 0
  fprintf('%s: finished with GPU device, resetting again\n', mfilename) ;
  gpuDevice(opts.gpus(1)) ;
end
fprintf('%s: all done\n', mfilename) ;
