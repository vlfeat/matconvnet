function info = cnn_imagenet_evaluate(varargin)
% CNN_IMAGENET_EVALUATE   Evauate MatConvNet models on ImageNet

run(fullfile(fileparts(mfilename('fullpath')), ...
  '..', 'matlab', 'vl_setupnn.m')) ;

opts.dataDir = fullfile('data', 'ILSVRC2012') ;
opts.expDir = fullfile('data', 'imagenet12-eval-vgg-f') ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.modelPath = fullfile('data', 'models', 'imagenet-vgg-f.mat') ;
opts.lite = false ;
opts.numFetchThreads = 12 ;
opts.train.batchSize = 128 ;
opts.train.numEpochs = 1 ;
opts.train.gpus = [] ;
opts.train.prefetch = true ;
opts.train.expDir = opts.expDir ;

opts = vl_argparse(opts, varargin) ;
display(opts);

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

net = load(opts.modelPath) ;
if isfield(net, 'net') ;
  net = net.net ;
end
isDag = isfield(net, 'vars') ;
if ~isDag
  net.classes = imdb.classes ;
  net.layers{end}.type = 'softmaxloss' ; % softmax -> softmaxloss
  net.normalization.border = [256 256] - net.normalization.imageSize(1:2) ;
  vl_simplenn_display(net, 'batchSize', opts.train.batchSize) ;

  % Synchronize label indexes between the model and the image database
  imdb = cnn_imagenet_sync_labels(imdb, net);

  bopts = net.normalization ;
  bopts.numThreads = opts.numFetchThreads ;
  fn = getBatchSimpleNNWrapper(bopts) ;

  [net,info] = cnn_train(net, imdb, fn, opts.train, ...
                         'train', NaN, ...
                         'val', find(imdb.images.set==2), ...
                         'conserveMemory', true) ;
else
  net = dagnn.DagNN.loadobj(net) ;

  bopts = net.meta.normalization ;
  bopts.numThreads = opts.numFetchThreads ;
  fn = getBatchDagNNWrapper(bopts, numel(opts.train.gpus) > 0) ;

  [net,info] = cnn_train_dag(net, imdb, fn, opts.train, ...
                             'train', NaN, ...
                             'val', find(imdb.images.set==2)) ;
end

% -------------------------------------------------------------------------
function fn = getBatchSimpleNNWrapper(opts)
% -------------------------------------------------------------------------
fn = @(imdb,batch) getBatchSimpleNN(imdb,batch,opts) ;

% -------------------------------------------------------------------------
function [im,labels] = getBatchSimpleNN(imdb, batch, opts)
% -------------------------------------------------------------------------
images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
im = cnn_imagenet_get_batch(images, opts, ...
                            'prefetch', nargout == 0) ;
labels = imdb.images.label(batch) ;

% -------------------------------------------------------------------------
function fn = getBatchDagNNWrapper(opts, useGpu)
% -------------------------------------------------------------------------
fn = @(imdb,batch) getBatchDagNN(imdb,batch,opts,useGpu) ;

% -------------------------------------------------------------------------
function inputs = getBatchDagNN(imdb, batch, opts, useGpu)
% -------------------------------------------------------------------------
images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
im = cnn_imagenet_get_batch(images, opts, ...
                            'prefetch', nargout == 0) ;
if nargout > 0
  if useGpu
    im = gpuArray(im) ;
  end
  inputs = {'input', im, 'label', imdb.images.label(batch)} ;
end
