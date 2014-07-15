function imagenet()
% CNN_IMAGENET   Demonstrates MatConvNet on ImageNet

opts.dataDir = 'data/imagenet12' ;
opts.expDir = 'data/imagenet12-exp-2' ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.lite = false ;
opts.train.batchSize = 100 ;
opts.train.numEpochs = 10 ;
opts.train.continue = true ;
opts.train.useGpu = true ;
opts.train.learningRate = [0.001*ones(1, 8) 0.0001*ones(1,2)] ;
opts.train.expDir = opts.expDir ;

run(fullfile(fileparts(mfilename('fullpath')), '../matlab/vl_setupnn.m')) ;

% -------------------------------------------------------------------------
%                                                   Database initialization
% -------------------------------------------------------------------------

if exist(opts.imdbPath)
  imdb = load(opts.imdbPath) ;
else
  imdb = cnn_imagenet_setup_data('dataDir', opts.dataDir, 'lite', opts.lite) ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end

% -------------------------------------------------------------------------
%                                                    Network initialization
% -------------------------------------------------------------------------

net = initializeNetwork(opts);
net.normalization.imageSize = [227, 227] ;

averageImagePath = fullfile(opts.expDir, 'average.mat') ;
if exist(averageImagePath)
  load(averageImagePath, 'averageImage') ;
else
  train = find(imdb.images.set == 1) ;
  bs = 256 ;
  for t=1:bs:numel(train)
    batch_time = tic ;
    batch = train(t:min(t+bs-1, numel(train))) ;
    fprintf('computing average image: processing batch starting with image %d ...', batch(1)) ;
    temp = getBatch(imdb, batch, ...
      'size', net.normalization.imageSize) ;
    im{t} = mean(temp, 4) ;
    batch_time = toc(batch_time) ;
    fprintf(' %.2f s (%.1f images/s)\n', batch_time, numel(batch)/ batch_time) ;
  end
  averageImage = mean(cat(4, im{:}),4) ;
  save(averageImagePath, 'averageImage') ;
end

net.normalization.averageImage = averageImage ;
clear averageImage im temp ;

% -------------------------------------------------------------------------
%                                               Stochastic gradient descent
% -------------------------------------------------------------------------

fn = getBatchWrapper(...
  net.normalization.averageImage, ...
  net.normalization.imageSize) ;

[net,info] = cnn_train(net, imdb, fn, opts.train) ;

% -------------------------------------------------------------------------
function fn = getBatchWrapper(averageImage, size)
% -------------------------------------------------------------------------
fn = @(imdb,batch) getBatch(imdb,batch,...
  'average',averageImage,...
  'size', size) ;

% -------------------------------------------------------------------------
function [im, labels] = getBatch(imdb, batch, varargin)
% -------------------------------------------------------------------------
opts.size = [227, 227] ;
opts.boder = [29, 29] ;
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
parfor i=1:numel(batch)
  imt = imread([imdb.imageDir '/' imdb.images.name{batch(i)}]) ;
  imt = single(imt) * (1/255) ; % faster than im2single
  if size(imt,3) == 1, imt = cat(3, imt, imt, imt) ; end
  w = size(imt,2) ;
  h = size(imt,1) ;

  fx = (opts.size(2)+opts.boder(2))/w ;
  fy = (opts.size(1)+opts.boder(1))/h ;
  factor = max(fx,fy) ;
  imt = imresize(imt, factor) ;
  
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

% -------------------------------------------------------------------------
function net = initializeNetwork(opt)
% -------------------------------------------------------------------------

scal = 1 ;

net.layers = {} ;

% Block 1
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', 0.01/scal * randn(11, 11, 3, 96, 'single'), ...
                           'biases', ones(1, 96, 'single'), ...
                           'stride', 4, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'relu') ;
net.layers{end+1} = struct('type', 'pool', ...
    'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'normalize', ...
                           'param', [5 1 0.0001/5 0.75], ...
                           'stride', 2, ...
                           'pad', 0) ;

% Block 2
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', 0.01/scal * randn(5, 5, 48, 256, 'single'), ...
                           'biases', ones(1, 256, 'single'), ...
                           'stride', 1, ...
                           'pad', 2) ;
net.layers{end+1} = struct('type', 'relu') ;
net.layers{end+1} = struct('type', 'pool', ...
  'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'normalize', ...
                           'param', [5 1 0.0001/5 0.75], ...
                           'stride', 2, ...
                           'pad', 0) ;

% Block 3
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', 0.01/scal * randn(3,3,256,384,'single'), ...
                           'biases', ones(1,384,'single'), ...
                           'stride', 1, ...
                           'pad', 1) ;
net.layers{end+1} = struct('type', 'relu') ;

% Block 4
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', 0.01/scal * randn(3,3,192,384,'single'), ...
                           'biases', ones(1,384,'single'), ...
                           'stride', 1, ...
                           'pad', 1) ;
net.layers{end+1} = struct('type', 'relu') ;

% Block 5
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', 0.01/scal * randn(3,3,192,256,'single'), ...
                           'biases', ones(1,256,'single'), ...
                           'stride', 1, ...
                           'pad', 1) ;
net.layers{end+1} = struct('type', 'relu') ;
net.layers{end+1} = struct('type', 'pool', ...
    'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', 0) ;

% Block 6
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', 0.01/scal * randn(6,6,256,4096,'single'),...
                           'biases', ones(1,4096,'single'), ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'relu') ;

% Block 7
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', 0.01/scal * randn(1,1,4096,4096,'single'),...
                           'biases', ones(1,4096,'single'), ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'relu') ;

% Block 8
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', 0.01/scal * randn(1,1,4096,1000,'single'), ...
                           'biases', zeros(1, 1000, 'single'), ...
                           'stride', 1, ...
                           'pad', 0) ;

% Block 9
net.layers{end+1} = struct('type', 'softmaxloss') ;
