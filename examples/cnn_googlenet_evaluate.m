function info = cnn_googlenet_evaluate(net, imdb, varargin)
% CNN_IMAGENET_EVALUATE   Evauate MatConvNet models on ImageNet

run(fullfile(fileparts(mfilename('fullpath')), ...
  '..', 'matlab', 'vl_setupnn.m')) ;

opts.dataDir = fullfile('data', 'imagenet12') ;
opts.expDir = fullfile('data', 'imagenet-eval-googlenet') ;
opts.lite = false ;
opts.numFetchThreads = 8 ;
opts.train.batchSize = 20 ;
opts.train.numEpochs = 1 ;
opts.train.useGpu = true ;
opts.train.prefetch = false ;
opts.train.expDir = opts.expDir ;

opts = vl_argparse(opts, varargin) ;
display(opts);

% -------------------------------------------------------------------------
%                                                    Network initialization
% -------------------------------------------------------------------------

% Synchronize label indexes between the model and the image database
imdb = cnn_imagenet_sync_labels(imdb, net);

% -------------------------------------------------------------------------
%                                               Stochastic gradient descent
% -------------------------------------------------------------------------

fn = getBatchWrapper(net.normalization, opts.numFetchThreads) ;

[net,info] = cnn_dagtrain(net, imdb, fn, opts.train, ...
  'conserveMemory', true, ...
  'train', NaN, ...
  'val', find(imdb.images.set==2)) ;

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
                            'prefetch', nargout == 0) ;
labels = imdb.images.label(batch) ;
