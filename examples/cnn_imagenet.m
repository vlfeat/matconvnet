function cnn_imagenet(varargin)
% CNN_IMAGENET   Demonstrates training a CNN on ImageNet

run(fullfile(fileparts(mfilename('fullpath')), ...
  '..', 'matlab', 'vl_setupnn.m')) ;

opts.dataDir = fullfile('data','ILSVRC2012') ;
opts.modelType = 'dropout' ; % bnorm or dropout
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.expDir = fullfile('data', sprintf('imagenet12-%s', opts.modelType)) ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.numFetchThreads = 12 ;
opts.lite = false ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.train.batchSize = 256 ;
opts.train.numSubBatches = 1 ;
opts.train.continue = true ;
opts.train.gpus = [] ;
opts.train.prefetch = false ;
opts.train.expDir = opts.expDir ;
switch opts.modelType
  case 'dropout'
    opts.train.learningRate = [0.01*ones(1,25) 0.001*ones(1,25) 0.0001*ones(1,15)] ;
    opts.train.learningRate = logspace(-2, -4, 75) ;
  case 'bnorm'
    opts.train.learningRate = [0.01*ones(1,5) 0.005*ones(1,5) 0.001*ones(1,5) 0.0001*ones(1,5)] ;
    opts.train.learningRate = logspace(-2, -4, 15) ;
end
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.train.numEpochs = numel(opts.train.learningRate) ;
opts = vl_argparse(opts, varargin) ;

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

switch opts.modelType
  case 'dropout'
    net = cnn_imagenet_init() ;
  case 'bnorm'
    net = cnn_imagenet_init_bnorm() ;
  otherwise
    error('Unknown model type %s', opts.modelType) ;
end

% compute the average image
averageImagePath = fullfile(opts.expDir, 'average.mat') ;
if exist(averageImagePath)
  load(averageImagePath, 'averageImage') ;
else
  train = find(imdb.images.set == 1) ;
  bs = 256 ;
  fn = getBatchWrapper(net.normalization, opts.numFetchThreads) ;
  for t=1:bs:numel(train)
    batch_time = tic ;
    batch = train(t:min(t+bs-1, numel(train))) ;
    fprintf('computing average image: processing batch starting with image %d ...', batch(1)) ;
    temp = fn(imdb, batch) ;
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

fn = getBatchWrapper(net.normalization, opts.numFetchThreads) ;

[net,info] = cnn_train(net, imdb, fn, opts.train, 'conserveMemory', true) ;

% -------------------------------------------------------------------------
function fn = getBatchWrapper(opts, numThreads)
% -------------------------------------------------------------------------
fn = @(imdb,batch) getBatch(imdb,batch,opts,numThreads) ;

% -------------------------------------------------------------------------
function [im,labels] = getBatch(imdb, batch, opts, numThreads)
% -------------------------------------------------------------------------
images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
im = cnn_imagenet_get_batch(images, opts, ...
                            'numThreads', numThreads, ...
                            'prefetch', nargout == 0, ...
                            'augmentation', 'f25') ;
labels = imdb.images.label(batch) ;


