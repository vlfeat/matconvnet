function imo = cnn_imagenet_get_batch(images, varargin)
% CNN_IMAGENET_GET_BATCH  Load, preprocess, and pack images for CNN evaluation

opts.imageSize = [227, 227] ;
opts.border = [29, 29] ;
opts.averageImage = [] ;
opts.augmentation = 'none' ;
opts.interpolation = 'bilinear' ;
opts.numAugments = 1 ;
opts.numThreads = 0 ;
opts.prefetch = false ;
opts.keepAspect = true;
opts = vl_argparse(opts, varargin);

% fetch is true if images is a list of filenames (instead of
% a cell array of images)
fetch = numel(images) > 1 && ischar(images{1}) ;

% prefetch is used to load images in a separate thread
prefetch = fetch & opts.prefetch ;

switch opts.augmentation
  case 'none'
    tfs = [.5 ; .5 ; 0 ];
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
  im = images ;
end

imo = zeros(opts.imageSize(1), opts.imageSize(2), 3, ...
            numel(images)*opts.numAugments, 'single') ;

[~,augmentations] = sort(rand(size(tfs,2), numel(images)), 1) ;

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
    t = augmentations(ai,i) ;
    tf = tfs(:,t) ;
    dx = floor((w - opts.imageSize(2)) * tf(2)) ;
    dy = floor((h - opts.imageSize(1)) * tf(1)) ;
    sx = (1:opts.imageSize(2)) + dx ;
    sy = (1:opts.imageSize(1)) + dy ;
    if tf(3), sx = fliplr(sx) ; end
    imo(:,:,:,si) = imt(sy,sx,:) ;
    si = si + 1 ;
  end
end

if ~isempty(opts.averageImage)
  imo = bsxfun(@minus, imo, opts.averageImage) ;
end
