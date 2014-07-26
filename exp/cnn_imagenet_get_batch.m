function [imo, labels] = cnn_imagenet_get_batch(imdb, batch, varargin)
% CNN_IMAGENET_GET_BATCH
opts.size = [227, 227] ;
opts.border = [29, 29] ;
opts.average = [] ;
opts.augmentation = 'none' ;
opts.numAugments = 1 ;
opts.numThreads = 0 ;
opts = vl_argparse(opts, varargin);

prefetch = nargout == 0 ;

switch opts.augmentation
  case 'none'
    tfs = [.5 ; .5 ; 0 ];
  case 'f5'
    tfs = [...
      .5 0 0 1 1 .5 0 1 0 1 ;
      .5 0 1 0 1 .5 0 1 0 1 ;
       0 0 0 0 0  1 1 1 1 1] ;
end

names = strcat([imdb.imageDir '/'], imdb.images.name(batch)) ;
im = cell(1, numel(batch)) ;
if opts.numThreads > 0
  if prefetch
    vl_imreadjpeg(names,'numThreads', opts.numThreads, 'prefetch') ;
    return ;
  end
  im = vl_imreadjpeg(names,'numThreads', opts.numThreads) ;
end

imo = zeros(opts.size(1), opts.size(2), 3, ...
  numel(batch)*opts.numAugments, 'single') ;
si = 1;
for i=1:numel(batch)
  if isempty(im{i})
    imt = imread(names{i}) ;
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
  if abs(factor - 1) > 0.01
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
    imo(:,:,:,si) = imt(1+dy:opts.size(1)+dy, ...
                       1+dx:opts.size(2)+dx, :) ;
    si = si + 1;
  end
end

if ~isempty(opts.average)
  imo = bsxfun(@minus, imo, opts.average);
end

labels = imdb.images.label(batch) ;