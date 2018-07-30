function [net, info] = cnn_office(varargin)
%CNN_OFFICE   Demonstrates training a Domain Adaptive Hash on Office and
%OfficeHome datasets

% Based on code from Andrea Vedaldi
% Copyright (C) 2016-17 Hemanth Venkateswara.
% All rights reserved.

run(fullfile(fileparts(mfilename('fullpath')), ...
  '..', '..', 'matlab', 'vl_setupnn.m')) ;

% Change name of dataset
% Office dataset domains: {'amazon', 'dslr', 'webcam'}
% OfficeHome dataset domains: {'Art', 'Clipart', 'Product', 'RealWorld'}
% Change the domains below to conduct the experiment
srcDataset = 'Product';
tgtDataset = 'Clipart';

opts.isOfficeHome = true;
if opts.isOfficeHome
    home = 'OfficeHome'; % Office or OfficeHome
    opts.imagesSubDir = '';
    opts.C = 65; % number of categories
else
    home = 'Office'; % Office or OfficeHome
    opts.imagesSubDir = 'images';
    opts.C = 31; % number of categories
end
opts.modelType = 'vgg-dah' ;
% Pretrained network path
preModelPath = '/data/DB/Office/imagenet-vgg-f.mat'; 
% Experimental results are stored at exp_root
exp_root = ['/data/DB/', home];
% Train data path (data is stored at data_root)
data_root = ['/home/ASUAD/hkdv1/CodeRep/MatConvNet/matconvnet-1.0-beta20/examples/', home];
jointDir = [srcDataset, '_', tgtDataset];
opts.expDir = fullfile(exp_root, sprintf('%s-%s', jointDir, opts.modelType)) ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.srcDataDir = fullfile(data_root, srcDataset) ;
opts.tgtDataDir = fullfile(data_root, tgtDataset) ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.preModelPath = preModelPath;
opts.trainSizePerCat = 20; % Not used for DAH. Set for Validation study
opts.valSizePerCat = 10; % Not used for DAH. Set for validation study
opts.valSizeRatio = 0; % Change for Validation study

opts.network = [] ;
opts.networkType = 'simplenn' ;
opts.batchNormalization = false ;
opts.weightInitMethod = 'gaussian' ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.networkType = 'simplenn' ;
opts.numFetchThreads = 12 ;
opts.lite = false ;
opts.train = struct() ;

%da_hash parameters
opts.train.hashSize = 64;
opts.train.gpus = 1;
opts.train.K = 5; % Number of samples per category
opts.train.gamma = 5*1e3; % Weight for linearMMD Loss
% opts.train.gamma = 1e2; % Weight for unbiasedMMD Loss
opts.train.l1 = 1.0; % Weight for hash loss
opts.train.entpW = 1.0; % Weight for entropy loss
opts.train.beta = 1.0; % Weight for Euclidean loss
opts.train.supHashW = 10.0; % Weight for supervised positive samples
opts = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;

% -------------------------------------------------------------------------
%                                                              Prepare data
% -------------------------------------------------------------------------

if exist(opts.imdbPath, 'file')
  imdb = load(opts.imdbPath) ;
else
  imdb = createDAHImdb(opts) ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end

% Compute image statistics (mean, RGB covariances, etc.)
fprintf(['Domain adaptation experiment with \nSource (%s) -> Target (%s)\nK = %d\n',...
    'Hash Size = %d\nHash Loss Weight = %0.3f\nSupHash Loss Weight = %0.3f\nEntropy Loss Weight = %0.3f\n', ...
    'Euclidean Loss = %0.3f\nMMD Loss Weight = %0.3f\n'], srcDataset, tgtDataset, ...
    opts.train.K, opts.train.hashSize, opts.train.l1, opts.train.supHashW, opts.train.entpW, ...
    opts.train.beta, opts.train.gamma);
imageStatsPath = fullfile(opts.expDir, 'imageStats.mat') ;
if exist(imageStatsPath)
  load(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance') ;
else
  train = find(imdb.images.set == 1) ;
  images = imdb.images.trainValNames(train);
  [averageImage, rgbMean, rgbCovariance] = getImageStats(images, ...
                                                    'imageSize', [224 224], ...
                                                    'numThreads', opts.numFetchThreads, ...
                                                    'gpus', opts.train.gpus) ;
  save(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance') ;
end
[v,d] = eig(rgbCovariance) ;
rgbDeviation = v*sqrt(d) ;
clear v d ;

% -------------------------------------------------------------------------
%                                                             Prepare model
% -------------------------------------------------------------------------

if isempty(opts.network)
  switch opts.modelType
    case 'resnet-50'
      net = cnn_imagenet_init_resnet('averageImage', rgbMean, ...
                                     'colorDeviation', rgbDeviation, ...
                                     'classNames', imdb.classes.name, ...
                                     'classDescriptions', imdb.classes.description) ;
      opts.networkType = 'dagnn' ;

    otherwise
      net = cnn_office_init('model', opts.modelType, ...
                              'batchNormalization', opts.batchNormalization, ...
                              'weightInitMethod', opts.weightInitMethod, ...
                              'networkType', opts.networkType, ...
                              'averageImage', rgbMean, ...
                              'colorDeviation', rgbDeviation, ...
                              'classNames', imdb.classes.name, ...
                              'classDescriptions', imdb.classes.description, ...
                              'preModelPath', opts.preModelPath, ...
                              'K', opts.train.K, ...
                              'gamma', opts.train.gamma, ...
                              'l1', opts.train.l1, ...
                              'entpW', opts.train.entpW, ...
                              'beta', opts.train.beta, ...
                              'supHashW', opts.train.supHashW, ...
                              'hashSize', opts.train.hashSize, ...
                              'C', opts.C) ;
  end
else
  net = opts.network ;
  opts.network = [] ;
end


% -------------------------------------------------------------------------
%                                                                     Learn
% -------------------------------------------------------------------------

switch opts.networkType
  case 'simplenn', trainFn = @cnn_train_dah ;
  case 'dagnn', trainFn = @cnn_train_dag ;
end

[net, info] = trainFn(net, imdb, getBatchFn(opts, net.meta), ...
                      'expDir', opts.expDir, ...
                      net.meta.trainOpts, ...
                      opts.train) ;

% -------------------------------------------------------------------------
%                                                                    Deploy
% -------------------------------------------------------------------------

% net = cnn_imagenet_deploy(net) ;
% modelPath = fullfile(opts.expDir, 'net-deployed.mat')
% 
% switch opts.networkType
%   case 'simplenn'
%     save(modelPath, '-struct', 'net') ;
%   case 'dagnn'
%     net_ = net.saveobj() ;
%     save(modelPath, '-struct', 'net_') ;
%     clear net_ ;
% end

% -------------------------------------------------------------------------
function fn = getBatchFn(opts, meta)
% -------------------------------------------------------------------------

if numel(meta.normalization.averageImage) == 3
  mu = double(meta.normalization.averageImage(:)) ;
else
  mu = imresize(single(meta.normalization.averageImage), ...
                meta.normalization.imageSize(1:2)) ;
end

useGpu = numel(opts.train.gpus) > 0 ;

bopts.test = struct(...
  'useGpu', useGpu, ...
  'numThreads', opts.numFetchThreads, ...
  'imageSize',  meta.normalization.imageSize(1:2), ...
  'cropSize', meta.normalization.cropSize, ...
  'subtractAverage', mu) ;

% Copy the parameters for data augmentation
bopts.train = bopts.test ;
for f = fieldnames(meta.augmentation)'
  f = char(f) ;
  bopts.train.(f) = meta.augmentation.(f) ;
end

fn = @(x,y) getBatch(bopts,useGpu,lower(opts.networkType),x,y) ;

% -------------------------------------------------------------------------
function varargout = getBatch(opts, useGpu, networkType, imdb, batch)
% -------------------------------------------------------------------------
% images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
images = imdb.images.trainValNames(batch);
if ~isempty(batch) && imdb.images.set(batch(1)) == 1
  phase = 'train' ;
else
  phase = 'test' ;
end
data = getImageBatch(images, opts.(phase), 'prefetch', nargout == 0) ;
if nargout > 0
  labels = imdb.images.label(batch) ;
  switch networkType
    case 'simplenn'
      varargout = {data, labels} ;
    case 'dagnn'
      varargout{1} = {'input', data, 'label', labels} ;
  end
end