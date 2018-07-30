function data = getImageBatch(imagePaths, varargin)
% GETIMAGEBATCH  Load and jitter a batch of images

opts.useGpu = false ;
opts.prefetch = false ;
opts.numThreads = 1 ;

opts.imageSize = [28, 28] ; % Changed from [227,227]
opts.cropSize = 28 / 28 ; % Changed from 227 / 256
opts.keepAspect = true ;
opts.subtractAverage = [] ;

opts.jitterFlip = false ;
opts.jitterLocation = false ;
opts.jitterAspect = 1 ;
opts.jitterScale = 1 ;
opts.jitterBrightness = 0 ;
opts.jitterContrast = 0 ;
opts.jitterSaturation = 0 ;

opts = vl_argparse(opts, varargin);

args{1} = {imagePaths, ...
           'NumThreads', opts.numThreads, ...
           'Pack', ...
           'Interpolation', 'bicubic', ...
           'Resize', opts.imageSize(1:2), ...
           'CropSize', opts.cropSize * opts.jitterScale, ...
           'CropAnisotropy', opts.jitterAspect, ...
           'Brightness', opts.jitterBrightness, ...
           'Contrast', opts.jitterContrast, ...
           'Saturation', opts.jitterSaturation} ;

if ~opts.keepAspect
  % Squashign effect
  args{end+1} = {'CropAnisotropy', 0} ;
end

if opts.jitterFlip
  args{end+1} = {'Flip'} ;
end

if opts.jitterLocation
  args{end+1} = {'CropLocation', 'random'} ;
else
  args{end+1} = {'CropLocation', 'center'} ;
end

if opts.useGpu
  args{end+1} = {'Gpu'} ;
end

if ~isempty(opts.subtractAverage)
  args{end+1} = {'SubtractAverage', opts.subtractAverage} ;
end

args = horzcat(args{:}) ;

if opts.prefetch
  vl_imreadjpeg(args{:}, 'prefetch') ;
  data = [] ;
else
  data = vl_imreadjpeg(args{:}) ;
  data = data{1} ;
end

% 'Pack' option in vl_imreadjpeg converts grayscale image to color by
% repeating. So, we will drop the redundant channels
n = size(data, 4);
data = reshape(shiftdim(data,2),3,[]) ; %[1 2 3 4] -> [3, 4, 1, 2] -> [3 x []]
data(2:3,:) = []; % drop redundant rows.
data = reshape(data, 1, n, opts.imageSize(1),opts.imageSize(2));
data = shiftdim(data,2);
