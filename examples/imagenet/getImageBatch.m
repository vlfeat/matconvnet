function data = getImageBatch(imagePaths, varargin)
% GETIMAGEBATCH  Load and jitter a batch of images

opts.fullImageSize = 256 ;
opts.imageSize = [227, 227] ;
opts.numThreads = 1 ;
opts.averageImage = [] ;
opts.useGpu = false ;
opts.prefetch = false ;
opts.jitter = false ;
opts.jitterAspect = 4/3 ;
opts.jitterScale = 1.2 ;
opts.jitterLight = 0.1 ;
opts.jitterBrightness = .4 ;
opts.jitterSaturation = .4 ;
opts.jitterContrast = .4 ;
opts.rgbSqrtCovariance = zeros(3,'single') ;
opts = vl_argparse(opts, varargin);

args = {imagePaths, ...
        'NumThreads', opts.numThreads, ...
        'Resize', opts.imageSize(1:2), ...
        'Pack'} ;

if opts.useGpu, args{end+1} = 'Gpu' ; end

if ~isempty(opts.averageImage)
  args = horzcat(args, ...
                 'SubtractAverage', double(opts.averageImage)) ;
end

if opts.jitter
  args = horzcat(args, ...
                 {'NumThreads', opts.numThreads, ...
                  'Resize', opts.imageSize(1:2), ...
                  'CropLocation', 'random', ...
                  'CropSize', max(opts.imageSize(1:2))/opts.fullImageSize * [1 1], ...
                  'CropAnisotropy', [1/opts.jitterAspect,opts.jitterAspect], ...
                  'Brightness', opts.jitterLight * double(opts.rgbSqrtCovariance), ...
                  'Saturation', opts.jitterSaturation, ...
                  'Contrast', opts.jitterContrast, ...
                  'Flip'}) ;
else
  args = horzcat(args, ...
                 {'CropLocation', 'center', ...
                  'CropSize', max(opts.imageSize(1:2))/opts.fullImageSize * [1 1]}) ;
end

if opts.prefetch
  vl_imreadjpeg2(args{:}, 'prefetch') ;
  data = [] ;
else
  data = vl_imreadjpeg2(args{:}) ;
  data = data{1} ;
end
