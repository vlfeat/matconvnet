function imo = cnn_imagenet_get_batch(images, varargin)
% CNN_IMAGENET_GET_BATCH  Load, preprocess, and pack images for CNN evaluation

opts.imageSize = [227, 227] ;
opts.border = [29, 29] ;
opts.keepAspect = true ;
opts.numAugments = 1 ;
opts.transformation = 'none' ;
opts.averageImage = [] ;
opts.rgbVariance = zeros(0,3,'single') ;
opts.interpolation = 'bilinear' ;
opts.numThreads = 1 ;
opts.prefetch = false ;
opts = vl_argparse(opts, varargin);

% fetch is true if images is a list of filenames (instead of
% a cell array of images)
fetch = numel(images) >= 1 && ischar(images{1}) ;

% prefetch is used to load images in a separate thread
prefetch = fetch & opts.prefetch ;

if prefetch
  vl_imreadjpeg(images, 'numThreads', opts.numThreads, 'prefetch') ;
  imo = [] ;
  return ;
end
if fetch
  im = vl_imreadjpeg(images,'numThreads', opts.numThreads) ;
else
  im = images ;
end

tfs = [] ;
switch opts.transformation
  case 'none'
    tfs = [
      .5 ;
      .5 ;
       0 ] ;
  case 'f5'
    tfs = [...
      .5 0 0 1 1 .5 0 0 1 1 ;
      .5 0 1 0 1 .5 0 1 0 1 ;
       0 0 0 0 0  1 1 1 1 1] ;
  case 'f25'
    [tx,ty] = meshgrid(linspace(0,1,5)) ;
    tfs = [tx(:)' ; ty(:)' ; zeros(1,numel(tx))] ;
    tfs_ = tfs ;
    tfs_(3,:) = 1 ;
    tfs = [tfs,tfs_] ;
  case 'stretch'
  otherwise
    error('Uknown transformations %s', opts.transformation) ;
end
[~,transformations] = sort(rand(size(tfs,2), numel(images)), 1) ;

if ~isempty(opts.rgbVariance) && isempty(opts.averageImage)
  opts.averageImage = zeros(1,1,3) ;
end
if numel(opts.averageImage) == 3
  opts.averageImage = reshape(opts.averageImage, 1,1,3) ;
end

imo = zeros(opts.imageSize(1), opts.imageSize(2), 3, ...
            numel(images)*opts.numAugments, 'single') ;

si = 1 ;
for i=1:numel(images)

  % acquire image
  if isempty(im{i})
    imt = imread(images{i}) ;
    imt = single(imt) ; % faster than im2single (and multiplies by 255)
  else
    imt = im{i} ;
  end
  if size(imt,3) == 1
    imt = cat(3, imt, imt, imt) ;
  end

  % resize
  w = size(imt,2) ;
  h = size(imt,1) ;
  factor = [(opts.imageSize(1)+opts.border(1))/h ...
            (opts.imageSize(2)+opts.border(2))/w];

  if opts.keepAspect
    factor = max(factor) ;
  end
  if any(abs(factor - 1) > 0.0001)
    imt = imresize(imt, ...
                   'scale', factor, ...
                   'method', opts.interpolation) ;
  end

  % crop & flip
  w = size(imt,2) ;
  h = size(imt,1) ;
  for ai = 1:opts.numAugments
    switch opts.transformation
      case 'stretch'
        sz = round(min(opts.imageSize(1:2)' .* (1-0.1+0.2*rand(2,1)), [w;h])) ;
        dx = randi(w - sz(2) + 1, 1) ;
        dy = randi(h - sz(1) + 1, 1) ;
        flip = rand > 0.5 ;
      otherwise
        tf = tfs(:, transformations(mod(ai-1, numel(transformations)) + 1)) ;
        sz = opts.imageSize(1:2) ;
        dx = floor((w - sz(2)) * tf(2)) + 1 ;
        dy = floor((h - sz(1)) * tf(1)) + 1 ;
        flip = tf(3) ;
    end
    sx = round(linspace(dx, sz(2)+dx-1, opts.imageSize(2))) ;
    sy = round(linspace(dy, sz(1)+dy-1, opts.imageSize(1))) ;
    if flip, sx = fliplr(sx) ; end

    if ~isempty(opts.averageImage)
      offset = opts.averageImage ;
      if ~isempty(opts.rgbVariance)
        offset = bsxfun(@plus, offset, reshape(opts.rgbVariance * randn(3,1), 1,1,3)) ;
      end
      imo(:,:,:,si) = bsxfun(@minus, imt(sy,sx,:), offset) ;
    else
      imo(:,:,:,si) = imt(sy,sx,:) ;
    end
    si = si + 1 ;
  end
end
