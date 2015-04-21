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
opts.train.conserveMemory = true ;
opts.train.sync = true ;

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
  net.classes = imdb.classes ;
end
net.layers{end}.type = 'softmaxloss' ; % softmax -> softmaxloss
net.normalization.border = [256 256] - net.normalization.imageSize(1:2) ;
vl_simplenn_display(net, 'batchSize', opts.train.batchSize) ;

% Synchronize label indexes between the model and the image database
imdb = cnn_imagenet_sync_labels(imdb, net);

% -------------------------------------------------------------------------
%                                               Stochastic gradient descent
% -------------------------------------------------------------------------

bopts = net.normalization ;
bopts.numThreads = opts.numFetchThreads ;
fn = getBatchWrapper(bopts) ;

[net,info] = cnn_train(net, imdb, fn, opts.train, ...
                            'train', NaN, ...
                            'val', find(imdb.images.set==2)) ;

% -------------------------------------------------------------------------
function fn = getBatchWrapper(opts)
% -------------------------------------------------------------------------
fn = @(imdb,batch) getBatch(imdb,batch,opts) ;

% -------------------------------------------------------------------------
function [im,labels] = getBatch(imdb, batch, opts)
% -------------------------------------------------------------------------
images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
im = cnn_imagenet_get_batch(images, opts, ...
                            'prefetch', nargout == 0) ;
labels = imdb.images.label(batch) ;
