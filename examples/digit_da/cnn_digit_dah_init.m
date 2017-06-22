function net = cnn_digit_dah_init(varargin)
% CNN_OFFICE_INIT  Initialize a standard CNN for ImageNet

% Based on code from Andrea Vedaldi
% Copyright (C) 2016-17 Hemanth Venkateswara.
% All rights reserved.

opts.scale = 1 ;
opts.initBias = 0 ;
opts.weightDecay = 1 ;
opts.weightInitMethod = 'xavierimproved' ;
% opts.weightInitMethod = 'gaussian' ;
opts.model = 'alexnet' ;
opts.batchNormalization = false ;
opts.networkType = 'simplenn' ;
opts.cudnnWorkspaceLimit = 1024*1024*1204 ; % 1GB
opts.classNames = {} ;
opts.classDescriptions = {} ;
opts.averageImage = zeros(3,1) ;
opts.colorDeviation = zeros(3) ;

opts.preModelPath = 'imagenet-vgg-f.mat';
opts.K = 20;
opts.gamma = 1;
opts.l1 = 1;
opts.entpW = 1;
opts.beta = 1;
opts.supHashW = 1;
opts.hashSize = 64;
opts.C = 10;
opts = vl_argparse(opts, varargin) ;

rng('default');
rng(0) ;

f=1/100 ;
net.layers = {} ;
net.meta.normalization.imageSize = [28, 28, 1] ; % Pretrained_DomainAdaptiveHash
bs = 2*(opts.C * opts.K) ; % source = 10*K = 200 + target = 400
    
net.layers{end+1} = struct('name', 'conv1', 'type', 'conv', ...
                           'weights', {{init_weight(opts, 5, 5, 1, 20, 'single'), zeros(1, 20, 'single')}}, ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('name', 'relu1', 'type', 'relu') ;
net.layers{end+1} = struct('name', 'pool1', 'type', 'pool', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0) ;
net.layers{end+1} = struct('name', 'conv2', 'type', 'conv', ...
                           'weights', {{init_weight(opts, 5, 5, 20, 50, 'single'), zeros(1,50,'single')}}, ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('name', 'relu2', 'type', 'relu') ;
net.layers{end+1} = struct('name', 'pool2', 'type', 'pool', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0) ;
net.layers{end+1} = struct('name', 'fc3', 'type', 'conv', ...
                           'weights', {{init_weight(opts, 4, 4, 5, 500, 'single'), zeros(1,500,'single')}}, ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('name', 'relu3', 'type', 'relu') ;
net = add_lmmdLoss(net, opts, '3') ;
net.layers{end+1} = struct('name', 'fc4', 'type', 'conv', ...
                           'weights', {{init_weight(opts, 1, 1, 500, opts.hashSize, 'single'), zeros(1,opts.hashSize,'single')}}, ...
                           'stride', 1, ...
                           'pad', 0) ;
net = add_lmmdLoss(net, opts, '4') ;
net.layers{end+1} = struct('type', 'bnorm', 'name', 'bn4', ...
                             'weights', {{ones(opts.hashSize, 1, 'single'), zeros(opts.hashSize, 1, 'single'), zeros(opts.hashSize, 2, 'single')}}, ...
                             'learningRate', [2 1 0.05], ...
                             'weightDecay', [0 0]) ;
if strcmp(opts.model, 'vgg-dah')
    net.layers{end+1} = struct('type', 'tanh', 'name', 'tanh') ; 
    net.layers{end+1} = struct('type', 'hash_entropy_loss', 'name', 'loss') ;
    net.meta.trainOpts.errorFunction = 'hash_entropy' ; % changed from 'multiclass'
elseif strcmp(opts.model, 'vgg-daSfx')
    net.layers{end+1} = struct('type', 'softmaxloss_entropyloss') ;
    net.meta.trainOpts.errorFunction = 'multiclass' ;
end

lr = logspace(-4, -5, 40);

net.meta.inputSize = [28 28 1] ;
net.meta.trainOpts.learningRate = lr ;
net.meta.trainOpts.numEpochs = length(lr) ;
net.meta.trainOpts.batchSize = bs ;


net.meta.normalization.cropSize = net.meta.normalization.imageSize(1) / 28 ;
net.meta.normalization.averageImage = opts.averageImage ;
net.meta.classes.name = opts.classNames ;
net.meta.classes.description = opts.classDescriptions;
net.meta.augmentation.jitterLocation = false ;
net.meta.augmentation.jitterFlip = false ;
% net.meta.augmentation.jitterBrightness = double(0.1 * opts.colorDeviation) ;
% net.meta.augmentation.jitterAspect = [2/3, 3/2] ;

net.meta.trainOpts.weightDecay = 0.0005 ; % changed from 0.0005
net.meta.trainOpts.K = opts.K;
net.meta.trainOpts.gamma = opts.gamma;
net.meta.trainOpts.l1 = opts.l1;
net.meta.trainOpts.entpW = opts.entpW;
net.meta.trainOpts.beta = opts.beta;
net.meta.trainOpts.supHashW = opts.supHashW;

% Fill in default values
net = vl_simplenn_tidy(net) ;

% --------------------------------------------------------------------
function net = add_lmmdLoss(net, opts, id)
% --------------------------------------------------------------------
if ~opts.batchNormalization
    net.layers{end+1} = struct('type', 'lmmd', ...
        'name', sprintf('lmmd%s', id)) ;
end

% -------------------------------------------------------------------------
function weights = init_weight(opts, h, w, in, out, type)
% -------------------------------------------------------------------------
% See K. He, X. Zhang, S. Ren, and J. Sun. Delving deep into
% rectifiers: Surpassing human-level performance on imagenet
% classification. CoRR, (arXiv:1502.01852v1), 2015.

switch lower(opts.weightInitMethod)
  case 'gaussian'
    sc = 0.01/opts.scale ;
    weights = randn(h, w, in, out, type)*sc;
  case 'xavier'
    sc = sqrt(3/(h*w*in)) ;
    weights = (rand(h, w, in, out, type)*2 - 1)*sc ;
  case 'xavierimproved'
    sc = sqrt(2/(h*w*out)) ;
    weights = randn(h, w, in, out, type)*sc ;
  otherwise
    error('Unknown weight initialization method''%s''', opts.weightInitMethod) ;
end