function [im, labels] = cnn_imagenet_get_batch(imdb, batch, varargin)
% CNN_IMAGENET_GET_BATCH
opts.size = [227, 227] ;
opts.border = [29, 29] ;
opts.average = [] ;
opts.augmentation = 'none' ;
opts = vl_argparse(opts, varargin);

switch opts.augmentation
  case 'none'
    tfs = [.5 ; .5 ; 0 ];
  case 'f5'
    tfs = [...
      .5 0 0 1 1 .5 0 1 0 1 ;
      .5 0 1 0 1 .5 0 1 0 1 ;
       0 0 0 0 0  1 1 1 1 1] ;
end

im = cell(1, numel(batch)) ;
for i=1:numel(batch)
  imt = imread([imdb.imageDir '/' imdb.images.name{batch(i)}]) ;
  imt = single(imt) ; % faster than im2single (and multiplies bt 255)
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
  t = randi(size(tfs,2),1) ;
  tf = tfs(:,t) ;
  dx = floor((w - opts.size(2)) * tf(2)) ;
  dy = floor((h - opts.size(1)) * tf(1)) ;
  sx = (1:opts.size(2)) + dx ;
  sy = (1:opts.size(1)) + dy ;
  if tf(3), sx = fliplr(sx) ; end
  imt = imt(sy,sx,:) ;

  % apply network normalization
  if ~isempty(opts.average)
    imt = imt - opts.average ;
  end
  im{i} = imt ;
end

im = cat(4, im{:}) ;
labels = imdb.images.label(batch) ;