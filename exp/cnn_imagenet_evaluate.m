function info = cnn_imagenet_evaluate(varargin)
% CNN_IMAGENET   Demonstrates MatConvNet on ImageNet

run(fullfile(fileparts(mfilename('fullpath')), '../matlab/vl_setupnn.m')) ;

opts.dataDir = 'data/imagenet12' ;
opts.expDir = 'data/imagenet12-eval-vgg-f' ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.modelPath = 'data/models/imagenet-vgg-f.mat' ;
opts.lite = false ;
opts.numFetchThreads = 8 ;
opts.train.batchSize = 256 ;
opts.train.numEpochs = 1 ;
opts.train.useGpu = false ;
opts.train.prefetch = false ;
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

% patch
imdb.images.name = strrep(imdb.images.name, '.JPEG', '.jpg') ;
imdb.imageDir = fullfile(opts.dataDir, 'images') ;

% -------------------------------------------------------------------------
%                                                    Network initialization
% -------------------------------------------------------------------------

net = load(opts.modelPath) ;
net.layers{end}.type = 'softmaxloss' ; % softmax -> softmaxloss

% IMDB and the loaded network may use a different label ordering
% This fixes this issue
imdb = cnn_imagenet_sync_labels(imdb, net);

% -------------------------------------------------------------------------
%                                               Stochastic gradient descent
% -------------------------------------------------------------------------

fn = getBatchWrapper(...
  net.normalization.averageImage, ...
  net.normalization.imageSize, ...
  opts.numFetchThreads) ;

[net,info] = cnn_train(net, imdb, fn, opts.train, ...
  'conserveMemory', true, ...
  'train', NaN, ...
  'val', find(imdb.images.set==2)) ;

% -------------------------------------------------------------------------
function fn = getBatchWrapper(averageImage, size, numThreads)
% -------------------------------------------------------------------------
fn = @(imdb,batch) getBatch(imdb,batch,averageImage,size,numThreads) ;

% -------------------------------------------------------------------------
function [im,labels] = getBatch(imdb, batch, averageImage, size, numThreads)
% -------------------------------------------------------------------------
images = strcat([imdb.imageDir '/'], imdb.images.name(batch)) ;
im = cnn_imagenet_get_batch(images, ...
                            'average', averageImage,...
                            'size', size, ...
                            'border', [0 0], ...
                            'numThreads', numThreads, ...
                            'prefetch', nargout == 0) ;
labels = imdb.images.label(batch) ;
