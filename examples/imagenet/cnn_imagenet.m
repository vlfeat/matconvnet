function [net, info] = cnn_imagenet(varargin)
%CNN_IMAGENET   Demonstrates training a CNN on ImageNet
%  This demo demonstrates training the AlexNet, VGG-F, VGG-S, VGG-M,
%  VGG-VD-16, and VGG-VD-19 architectures on ImageNet data.

run(fullfile(fileparts(mfilename('fullpath')), ...
  '..', '..', 'matlab', 'vl_setupnn.m')) ;

opts.dataDir = fullfile(vl_rootnn, 'data','ILSVRC2012') ;
opts.modelType = 'alexnet' ;
opts.networkType = 'simplenn' ;
opts.batchNormalization = true ;
opts.weightInitMethod = 'gaussian' ;
[opts, varargin] = vl_argparse(opts, varargin) ;

sfx = opts.modelType ;
if opts.batchNormalization, sfx = [sfx '-bnorm'] ; end
sfx = [sfx '-' opts.networkType] ;
opts.expDir = fullfile(vl_rootnn, 'data', ['imagenet12-' sfx]) ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.numFetchThreads = 12 ;
opts.lite = false ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.train = struct() ;
opts = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;

% -------------------------------------------------------------------------
%                                                             Prepare model
% -------------------------------------------------------------------------

switch opts.modelType
  case 'resnet-50'
    net = cnn_imagenet_init_resnet() ;
    opts.networkType = 'dagnn' ;

  case 'inception'
    net = cnn_imagenet_init_inception() ;
    opts.networkType = 'dagnn' ;

  otherwise
    net = cnn_imagenet_init('model', opts.modelType, ...
                            'batchNormalization', opts.batchNormalization, ...
                            'weightInitMethod', opts.weightInitMethod, ...
                            'networkType', opts.networkType) ;
end

% -------------------------------------------------------------------------
%                                                              Prepare data
% -------------------------------------------------------------------------

if exist(opts.imdbPath)
  imdb = load(opts.imdbPath) ;
else
  imdb = cnn_imagenet_setup_data('dataDir', opts.dataDir, 'lite', opts.lite) ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end

% Compute image statistics (mean, RGB covariances, etc.)
imageStatsPath = fullfile(opts.expDir, 'imageStats.mat') ;
if exist(imageStatsPath)
  load(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance') ;
else
  train = find(imdb.images.set == 1) ;
  images = fullfile(imdb.imageDir, imdb.images.name(train(1:100:end))) ;
  [averageImage, rgbMean, rgbCovariance] = getImageStats(...
    images, 'imageSize', net.meta.normalization.imageSize) ;
  save(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance') ;
end

% Set the class names in the network
net.meta.classes.name = imdb.classes.name ;
net.meta.classes.description = imdb.classes.description ;

% Set average image
net.meta.normalization.averageImage = rgbMean ;

% Set data augmentation statistics
[v,d] = eig(rgbCovariance) ;
net.meta.augmentation.rgbSqrtCovariance = v*sqrt(d) ;
clear v d ;

% -------------------------------------------------------------------------
%                                                                     Learn
% -------------------------------------------------------------------------

switch opts.networkType
  case 'simplenn', trainFn = @cnn_train ;
  case 'dagnn', trainFn = @cnn_train_dag ;
end

[net, info] = trainFn(net, imdb, getBatchFn(opts, net.meta), ...
                      'expDir', opts.expDir, ...
                      net.meta.trainOpts, ...
                      opts.train) ;

% -------------------------------------------------------------------------
%                                                                    Deploy
% -------------------------------------------------------------------------

net = cnn_imagenet_deploy(net) ;
modelPath = fullfile(opts.expDir, 'net-deployed.mat')

switch opts.networkType
  case 'simplenn'
    save(modelPath, '-struct', 'net') ;
  case 'dagnn'
    net_ = net.saveobj() ;
    save(modelPath, '-struct', 'net_') ;
    clear net_ ;
end

% -------------------------------------------------------------------------
function fn = getBatchFn(opts, meta)
% -------------------------------------------------------------------------
useGpu = numel(opts.train.gpus) > 0 ;
bopts.numThreads = opts.numFetchThreads ;
bopts.fullImageSize = meta.normalization.fullImageSize ;
bopts.imageSize = meta.normalization.imageSize ;
bopts.averageImage = meta.normalization.averageImage ;
bopts.rgbSqrtCovariance = meta.augmentation.rgbSqrtCovariance ;
bopts.jitter = meta.augmentation.jitter ;
bopts.jitterLight = meta.augmentation.jitterLight ;
bopts.jitterBrightness = meta.augmentation.jitterBrightness ;
bopts.jitterSaturation = meta.augmentation.jitterSaturation ;
bopts.jitterContrast = meta.augmentation.jitterContrast ;
fn = @(x,y) getBatch(bopts,useGpu,lower(opts.networkType),x,y) ;

% -------------------------------------------------------------------------
function varargout = getBatch(opts, useGpu, networkType, imdb, batch)
% -------------------------------------------------------------------------
images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
isTrain = ~isempty(batch) && imdb.images.set(batch(1)) == 1 ;
data = getImageBatch(images, ...
                     opts, ...
                     'prefetch', nargout == 0, ...
                     'jitter', isTrain, ...
                     'useGpu', useGpu) ;
if nargout > 0
  labels = imdb.images.label(batch) ;
  switch networkType
    case 'simplenn'
      varargout = {data, labels} ;
    case 'dagnn'
      varargout{1} = {'input', data, 'label', labels} ;
  end
end
