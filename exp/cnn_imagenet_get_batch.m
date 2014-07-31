function imo = cnn_imagenet_get_batch(images, varargin)
% CNN_IMAGENET_GET_BATCH
opts.size = [227, 227] ;
opts.border = [29, 29] ;
opts.average = [] ;
opts.augmentation = 'none' ;
opts.numAugments = 1 ;
opts.numThreads = 0 ;
opts.prefetch = false ;
opts = vl_argparse(opts, varargin);

fetch = numel(images) > 1 && ischar(images{1}) ;
prefetch = fetch & opts.prefetch ;

switch opts.augmentation
  case 'none'
    tfs = [.5 ; .5 ; 0 ];
  case 'f5'
    tfs = [...
      .5 0 0 1 1 .5 0 1 0 1 ;
      .5 0 1 0 1 .5 0 1 0 1 ;
       0 0 0 0 0  1 1 1 1 1] ;
end

im = cell(1, numel(images)) ;
if opts.numThreads > 0
  if prefetch
    vl_imreadjpeg(images, 'numThreads', opts.numThreads, 'prefetch') ;
    imo = [] ;
    return ;
  end
  if fetch
    im = vl_imreadjpeg(images,'numThreads', opts.numThreads) ;
  end
end
if ~fetch
  im = cell(1, numel(images)) ;
end

imo = zeros(opts.size(1), opts.size(2), 3, numel(images)*opts.numAugments, 'single') ;
si = 1;
for i=1:numel(images)
  if isempty(im{i})
    imt = imread(images{i}) ;
    imt = single(imt) ; % faster than im2single (and multiplies by 255)
  else
    imt = im{i} ;% 255 ;
  end
  if size(imt,3) == 1, imt = cat(3, imt, imt, imt) ; end
  w = size(imt,2) ;
  h = size(imt,1) ;

  fx = (opts.size(2)+opts.border(2))/w ;
  fy = (opts.size(1)+opts.border(1))/h ;
  factor = max(fx,fy) ;
  if abs(factor - 1) > 0.0001
    imt = imresize(imt, factor, 'bilinear') ;
  end

  % crop & flip
  w = size(imt,2) ;
  h = size(imt,1) ;
  augmentations = randperm(size(tfs,2));
  for ai = 1:opts.numAugments
    t = augmentations(ai) ;
    tf = tfs(:,t) ;
    dx = floor((w - opts.size(2)) * tf(2)) ;
    dy = floor((h - opts.size(1)) * tf(1)) ;
    sx = (1:opts.size(2)) + dx ;
    sy = (1:opts.size(1)) + dy ;
    if tf(3), sx = fliplr(sx) ; end
    imo(:,:,:,si) = imt(sy,sx,:) ;
    si = si + 1;
  end
end

if ~isempty(opts.average)
  imo = bsxfun(@minus, imo, opts.average);
end
