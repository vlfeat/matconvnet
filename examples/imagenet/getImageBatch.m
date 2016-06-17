function data = getImageBatch(imagePaths, varargin)
% GETIMAGEBATCH  Load and jitter a batch of images

opts.fullImageSize = 256 ;
opts.imageSize = [227, 227] ;
opts.numThreads = 1 ;
opts.averageImage = [] ;
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

args = {imagePaths, 'numThreads', opts.numThreads, 'resize', opts.fullImageSize} ;

if opts.prefetch
  vl_imreadjpeg(args{:}, 'prefetch') ;
  data = [] ;
  return ;
end

data = zeros(opts.imageSize(1), opts.imageSize(2), 3, numel(imagePaths), 'single') ;
images = vl_imreadjpeg(args{:}) ;

for t = 1:numel(images)

  % get next image
  im = images{t} ;
  if size(im,3) == 1
    im = cat(3, im, im, im) ;
  end
  sourceSize = size(im) ;

  % crop and stretech
  if opts.jitter
    aspect = exp((2*rand-1) * log(opts.jitterAspect)) ;
    scale = exp((2*rand-1) * log(opts.jitterScale)) ;
    tw = opts.imageSize(2) * sqrt(aspect) * scale ;
    th = opts.imageSize(1) / sqrt(aspect) * scale ;
    reduce = min([sourceSize(2) / tw, sourceSize(1) / th, 1]) ;
    regionSize = round(reduce * [th ; tw]) ;
    dx = randi(sourceSize(2) - regionSize(2) + 1, 1) ;
    dy = randi(sourceSize(1) - regionSize(1) + 1, 1) ;
    flip = rand > 0.5 ;
  else
    regionSize = opts.imageSize ;
    dx = floor((sourceSize(2) - regionSize(2))/2) + 1 ;
    dy = floor((sourceSize(1) - regionSize(1))/2) + 1 ;
    flip = false ;
  end

  sx = round(linspace(dx, regionSize(2)+dx-1, opts.imageSize(2))) ;
  sy = round(linspace(dy, regionSize(1)+dy-1, opts.imageSize(1))) ;
  if flip, sx = fliplr(sx) ; end

  im = im(sy,sx,:) ;
  data(:,:,:,t) = jitterColors(opts, im) ;
end

% --------------------------------------------------------------------
function im = jitterColors(opts, im)
% --------------------------------------------------------------------

colorOffset = [] ;
grayOffset = [] ;
rgbOffset = [] ;

if numel(opts.averageImage) ~= 3
  colorOffset = opts.averageImage ;
else
  rgbOffset = reshape(opts.averageImage, 1,1,3) ;
end

%    sat * con * bri * (im - averageImage)
%  + sat * con * (1-bri) * zero
%  + sat * (1-con) * average
%  + (1-sat) * gray
%  + light

if opts.jitter
  bri = 1 + opts.jitterBrightness * (2*rand-1) ;
  sat = 1 + opts.jitterSaturation * (2*rand-1) ;
  con = 1 + opts.jitterContrast   * (2*rand-1) ;
  scale = sat * con * bri ;

  rgbOffset = scale * rgbOffset ;

  grayOffset = mean(im,3) ;
  average = mean(grayOffset(:)) ;
  grayOffset = (1-sat) * grayOffset ;
  rgbOffset = rgbOffset + (sat * (1-con)) * average ;
  rgbOffset = rgbOffset + opts.jitterLight * ...
      reshape(opts.rgbSqrtCovariance * randn(3,1), 1,1,3) ;
else
  scale = 1 ;
end

% color offset does not incorporate scale yet, so subtract if first
if ~isempty(colorOffset), im = im - colorOffset ; end
if scale ~= 1, im = scale * im ; end
if ~isempty(grayOffset), im = bsxfun(@minus, im, grayOffset) ; end
if ~isempty(rgbOffset), im = bsxfun(@minus, im, rgbOffset) ; end
