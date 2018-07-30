function net = cnn_office_init(varargin)
% CNN_OFFICE_INIT  Initialize a standard CNN for ImageNet
% The code is similar to cnn_imagenet_init. Have added a new hash network
% vgg-dah

% Based on code from Andrea Vedaldi
% Copyright (C) 2016-17 Hemanth Venkateswara.
% All rights reserved.

opts.scale = 1 ;
opts.initBias = 0 ;
opts.weightDecay = 1 ;
%opts.weightInitMethod = 'xavierimproved' ;
opts.weightInitMethod = 'gaussian' ;
opts.model = 'alexnet' ;
opts.batchNormalization = false ;
opts.networkType = 'simplenn' ;
opts.cudnnWorkspaceLimit = 1024*1024*1204 ; % 1GB
opts.classNames = {} ;
opts.classDescriptions = {} ;
opts.averageImage = zeros(3,1) ;
opts.colorDeviation = zeros(3) ;

opts.preModelPath = 'imagenet-vgg-f.mat';
opts.K = 5;
opts.gamma = 1;
opts.l1 = 1;
opts.entpW = 1;
opts.beta = 1;
opts.supHashW = 1;
opts.hashSize = 64;
opts.C = 31;
opts = vl_argparse(opts, varargin) ;

% Define layers
switch opts.model
  case 'alexnet'
    net.meta.normalization.imageSize = [227, 227, 3] ;
    net = alexnet(net, opts) ;
    bs = 256 ;
  case 'vgg-f'
    net.meta.normalization.imageSize = [224, 224, 3] ;
    net = vgg_f(net, opts) ;
    bs = 256 ;
  case {'vgg-m', 'vgg-m-1024'}
    net.meta.normalization.imageSize = [224, 224, 3] ;
    net = vgg_m(net, opts) ;
    bs = 196 ;
  case 'vgg-s'
    net.meta.normalization.imageSize = [224, 224, 3] ;
    net = vgg_s(net, opts) ;
    bs = 128 ;
  case 'vgg-vd-16'
    net.meta.normalization.imageSize = [224, 224, 3] ;
    net = vgg_vd(net, opts) ;
    bs = 32 ;
  case 'vgg-vd-19'
    net.meta.normalization.imageSize = [224, 224, 3] ;
    net = vgg_vd(net, opts) ;
    bs = 24 ;
  case 'vgg-dah'
    net.meta.normalization.imageSize = [224, 224, 3] ; % Pretrained_DomainAdaptiveHash
    net = vgg_dah(net, opts) ;
    bs = 2*(opts.C * opts.K) ; % source = 31*K = 155 + target = 155
  otherwise
    error('Unknown model ''%s''', opts.model) ;
end

% final touches
switch lower(opts.weightInitMethod)
  case {'xavier', 'xavierimproved'}
    net.layers{end}.weights{1} = net.layers{end}.weights{1} / 10 ;
end
if strcmp(opts.model, 'vgg-dah')
    net.layers{end+1} = struct('type', 'hash_entropy_loss', 'name', 'loss') ;
else
    net.layers{end+1} = struct('type', 'softmaxloss', 'name', 'loss') ;
end

% Meta parameters
net.meta.inputSize = [net.meta.normalization.imageSize, 32] ;
net.meta.normalization.cropSize = net.meta.normalization.imageSize(1) / 224 ;
net.meta.normalization.averageImage = opts.averageImage ;
net.meta.classes.name = opts.classNames ;
net.meta.classes.description = opts.classDescriptions;
net.meta.augmentation.jitterLocation = true ;
net.meta.augmentation.jitterFlip = true ;
net.meta.augmentation.jitterBrightness = double(0.1 * opts.colorDeviation) ;
net.meta.augmentation.jitterAspect = [2/3, 3/2] ;

if ~opts.batchNormalization
  % lr = logspace(-2, -4, 60) ;
  lr = logspace(-4, -5, 300) ; % 64 bits
else
  lr = logspace(-1, -4, 20) ;
end

net.meta.trainOpts.learningRate = lr ;
net.meta.trainOpts.numEpochs = numel(lr) ;
net.meta.trainOpts.batchSize = bs ;
net.meta.trainOpts.weightDecay = 0.0005 ;
net.meta.trainOpts.K = opts.K;
net.meta.trainOpts.gamma = opts.gamma;
net.meta.trainOpts.l1 = opts.l1;
net.meta.trainOpts.entpW = opts.entpW;
net.meta.trainOpts.beta = opts.beta;
net.meta.trainOpts.supHashW = opts.supHashW;

% Fill in default values
net = vl_simplenn_tidy(net) ;

% Switch to DagNN if requested
switch lower(opts.networkType)
  case 'simplenn'
    % done
  case 'dagnn'
    net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;
    net.addLayer('top1err', dagnn.Loss('loss', 'classerror'), ...
                 {'prediction','label'}, 'top1err') ;
    net.addLayer('top5err', dagnn.Loss('loss', 'topkerror', ...
                                       'opts', {'topK',5}), ...
                 {'prediction','label'}, 'top5err') ;
  otherwise
    assert(false) ;
end

% --------------------------------------------------------------------
function net = add_block(net, opts, id, h, w, in, out, stride, pad)
% --------------------------------------------------------------------
info = vl_simplenn_display(net) ;
fc = (h == info.dataSize(1,end) && w == info.dataSize(2,end)) ;
if fc
  name = 'fc' ;
else
  name = 'conv' ;
end
convOpts = {'CudnnWorkspaceLimit', opts.cudnnWorkspaceLimit} ;
net.layers{end+1} = struct('type', 'conv', 'name', sprintf('%s%s', name, id), ...
                           'weights', {{init_weight(opts, h, w, in, out, 'single'), ...
                             ones(out, 1, 'single')*opts.initBias}}, ...
                           'stride', stride, ...
                           'pad', pad, ...
                           'dilate', 1, ...
                           'learningRate', [1 2], ...
                           'weightDecay', [opts.weightDecay 0], ...
                           'opts', {convOpts}) ;
if opts.batchNormalization
  net.layers{end+1} = struct('type', 'bnorm', 'name', sprintf('bn%s',id), ...
                             'weights', {{ones(out, 1, 'single'), zeros(out, 1, 'single'), ...
                               zeros(out, 2, 'single')}}, ...
                             'epsilon', 1e-4, ...
                             'learningRate', [2 1 0.1], ...
                             'weightDecay', [0 0]) ;
end
if strcmp(opts.model, 'vgg-dah') && strcmp(id, '8')
    % bnorm after fc8 only since it is followed by tanh which can saturate
    % Add lmmdloss after fc8 also
   net = add_lmmdLoss(net, opts, id) ; 
    % Add bnorm
   net.layers{end+1} = struct('type', 'bnorm', 'name', sprintf('bn%s',id), ...
                             'weights', {{ones(out, 1, 'single'), zeros(out, 1, 'single'), zeros(out, 2, 'single')}}, ...
                             'learningRate', [2 1 0.05], ...
                             'weightDecay', [0 0]) ;
   net.layers{end+1} = struct('type', 'tanh', 'name', 'tanh') ; 

end
net.layers{end+1} = struct('type', 'relu', 'name', sprintf('relu%s',id)) ;

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

% --------------------------------------------------------------------
function net = add_norm(net, opts, id)
% --------------------------------------------------------------------
if ~opts.batchNormalization
  net.layers{end+1} = struct('type', 'normalize', ...
                             'name', sprintf('norm%s', id), ...
                             'param', [5 1 0.0001/5 0.75]) ;
end

% --------------------------------------------------------------------
function net = add_dropout(net, opts, id)
% --------------------------------------------------------------------
if ~opts.batchNormalization
  net.layers{end+1} = struct('type', 'dropout', ...
                             'name', sprintf('dropout%s', id), ...
                             'rate', 0.5) ;
end

% --------------------------------------------------------------------
function net = add_lmmdLoss(net, opts, id)
% --------------------------------------------------------------------
if ~opts.batchNormalization
    net.layers{end+1} = struct('type', 'lmmd', ...
        'name', sprintf('lmmd%s', id)) ;
end

% --------------------------------------------------------------------
function net = alexnet(net, opts)
% --------------------------------------------------------------------

net.layers = {} ;

net = add_block(net, opts, '1', 11, 11, 3, 96, 4, 0) ;
net = add_norm(net, opts, '1') ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool1', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', 0) ;


net = add_block(net, opts, '2', 5, 5, 48, 256, 1, 2) ;
net = add_norm(net, opts, '2') ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool2', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', 0) ;


net = add_block(net, opts, '3', 3, 3, 256, 384, 1, 1) ;
net = add_block(net, opts, '4', 3, 3, 192, 384, 1, 1) ;
net = add_block(net, opts, '5', 3, 3, 192, 256, 1, 1) ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool5', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', 0) ;

net = add_block(net, opts, '6', 6, 6, 256, 4096, 1, 0) ;
net = add_dropout(net, opts, '6') ;

net = add_block(net, opts, '7', 1, 1, 4096, 4096, 1, 0) ;
net = add_dropout(net, opts, '7') ;

net = add_block(net, opts, '8', 1, 1, 4096, 1000, 1, 0) ;
net.layers(end) = [] ;
if opts.batchNormalization, net.layers(end) = [] ; end

% --------------------------------------------------------------------
function net = vgg_s(net, opts)
% --------------------------------------------------------------------

net.layers = {} ;
net = add_block(net, opts, '1', 7, 7, 3, 96, 2, 0) ;
net = add_norm(net, opts, '1') ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool1', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 3, ...
                           'pad', [0 2 0 2]) ;

net = add_block(net, opts, '2', 5, 5, 96, 256, 1, 0) ;
net = add_norm(net, opts, '2') ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool2', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', [0 1 0 1]) ;

net = add_block(net, opts, '3', 3, 3, 256, 512, 1, 1) ;
net = add_block(net, opts, '4', 3, 3, 512, 512, 1, 1) ;
net = add_block(net, opts, '5', 3, 3, 512, 512, 1, 1) ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool5', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 3, ...
                           'pad', [0 1 0 1]) ;

net = add_block(net, opts, '6', 6, 6, 512, 4096, 1, 0) ;
net = add_dropout(net, opts, '6') ;

net = add_block(net, opts, '7', 1, 1, 4096, 4096, 1, 0) ;
net = add_dropout(net, opts, '7') ;

net = add_block(net, opts, '8', 1, 1, 4096, 1000, 1, 0) ;
net.layers(end) = [] ;
if opts.batchNormalization, net.layers(end) = [] ; end

% --------------------------------------------------------------------
function net = vgg_m(net, opts)
% --------------------------------------------------------------------

net.layers = {} ;
net = add_block(net, opts, '1', 7, 7, 3, 96, 2, 0) ;
net = add_norm(net, opts, '1') ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool1', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', 0) ;

net = add_block(net, opts, '2', 5, 5, 96, 256, 2, 1) ;
net = add_norm(net, opts, '2') ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool2', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', [0 1 0 1]) ;

net = add_block(net, opts, '3', 3, 3, 256, 512, 1, 1) ;
net = add_block(net, opts, '4', 3, 3, 512, 512, 1, 1) ;
net = add_block(net, opts, '5', 3, 3, 512, 512, 1, 1) ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool5', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', 0) ;

net = add_block(net, opts, '6', 6, 6, 512, 4096, 1, 0) ;
net = add_dropout(net, opts, '6') ;

switch opts.model
  case 'vgg-m'
    bottleneck = 4096 ;
  case 'vgg-m-1024'
    bottleneck = 1024 ;
end
net = add_block(net, opts, '7', 1, 1, 4096, bottleneck, 1, 0) ;
net = add_dropout(net, opts, '7') ;

net = add_block(net, opts, '8', 1, 1, bottleneck, 1000, 1, 0) ;
net.layers(end) = [] ;
if opts.batchNormalization, net.layers(end) = [] ; end

% --------------------------------------------------------------------
function net = vgg_f(net, opts)
% --------------------------------------------------------------------

net.layers = {} ;
net = add_block(net, opts, '1', 11, 11, 3, 64, 4, 0) ;
net = add_norm(net, opts, '1') ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool1', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', [0 1 0 1]) ;

net = add_block(net, opts, '2', 5, 5, 64, 256, 1, 2) ;
net = add_norm(net, opts, '2') ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool2', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', 0) ;

net = add_block(net, opts, '3', 3, 3, 256, 256, 1, 1) ;
net = add_block(net, opts, '4', 3, 3, 256, 256, 1, 1) ;
net = add_block(net, opts, '5', 3, 3, 256, 256, 1, 1) ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool5', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', 0) ;

net = add_block(net, opts, '6', 6, 6, 256, 4096, 1, 0) ;
net = add_dropout(net, opts, '6') ;

net = add_block(net, opts, '7', 1, 1, 4096, 4096, 1, 0) ;
net = add_dropout(net, opts, '7') ;

net = add_block(net, opts, '8', 1, 1, 4096, 1000, 1, 0) ;
net.layers(end) = [] ;
if opts.batchNormalization, net.layers(end) = [] ; end

% --------------------------------------------------------------------
function net = vgg_vd(net, opts)
% --------------------------------------------------------------------

net.layers = {} ;
net = add_block(net, opts, '1_1', 3, 3, 3, 64, 1, 1) ;
net = add_block(net, opts, '1_2', 3, 3, 64, 64, 1, 1) ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool1', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0) ;

net = add_block(net, opts, '2_1', 3, 3, 64, 128, 1, 1) ;
net = add_block(net, opts, '2_2', 3, 3, 128, 128, 1, 1) ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool2', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0) ;

net = add_block(net, opts, '3_1', 3, 3, 128, 256, 1, 1) ;
net = add_block(net, opts, '3_2', 3, 3, 256, 256, 1, 1) ;
net = add_block(net, opts, '3_3', 3, 3, 256, 256, 1, 1) ;
if strcmp(opts.model, 'vgg-vd-19')
  net = add_block(net, opts, '3_4', 3, 3, 256, 256, 1, 1) ;
end
net.layers{end+1} = struct('type', 'pool', 'name', 'pool3', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0) ;

net = add_block(net, opts, '4_1', 3, 3, 256, 512, 1, 1) ;
net = add_block(net, opts, '4_2', 3, 3, 512, 512, 1, 1) ;
net = add_block(net, opts, '4_3', 3, 3, 512, 512, 1, 1) ;
if strcmp(opts.model, 'vgg-vd-19')
  net = add_block(net, opts, '4_4', 3, 3, 512, 512, 1, 1) ;
end
net.layers{end+1} = struct('type', 'pool', 'name', 'pool4', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0) ;

net = add_block(net, opts, '5_1', 3, 3, 512, 512, 1, 1) ;
net = add_block(net, opts, '5_2', 3, 3, 512, 512, 1, 1) ;
net = add_block(net, opts, '5_3', 3, 3, 512, 512, 1, 1) ;
if strcmp(opts.model, 'vgg-vd-19')
  net = add_block(net, opts, '5_4', 3, 3, 512, 512, 1, 1) ;
end
net.layers{end+1} = struct('type', 'pool', 'name', 'pool5', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0) ;

net = add_block(net, opts, '6', 7, 7, 512, 4096, 1, 0) ;
net = add_dropout(net, opts, '6') ;

net = add_block(net, opts, '7', 1, 1, 4096, 4096, 1, 0) ;
net = add_dropout(net, opts, '7') ;

net = add_block(net, opts, '8', 1, 1, 4096, 1000, 1, 0) ;
net.layers(end) = [] ;
if opts.batchNormalization, net.layers(end) = [] ; end

% --------------------------------------------------------------------
function net = vgg_dah(net, opts) % Pretrained_DomainAdaptiveHash
% --------------------------------------------------------------------
% 1.conv1(11x11x3x64) -> 2.relu1 -> 3.norm1(lrn) -> 4.pool1(3x3, stride 2)
% 5.conv2(5x5x64x256) -> 6.relu2 -> 7.norm2(lrn) -> 8.pool2(3x3, stride 2)
% 9.conv3(3x3x256x256) -> 10.relu3
% 11.conv4(3x3x256x256) -> 12.relu4
% 13.conv5(3x3x256x256) -> 14.relu5 -> 15.pool5(3x3, stride 2)
% 16.fc6(6x6x256x4097) -> 17.relu6 -> 18.lmmd6
% 19.fc7(1x1x4096x4096) -> 20.relu7 -> 21.lmmd7
% 22.fc8(1x1x4096x64) -> 23.lmmd8 -> 24.bnorm -> 25.tanh -> 26.hash_entropy_loss

convOpts = {'CudnnWorkspaceLimit', opts.cudnnWorkspaceLimit} ;
fnet = load(opts.preModelPath);
n = numel(fnet.layers) ;
net.layers = {} ;
prevConv = '';
for ii = 1:n
    l = fnet.layers{ii} ;
    if strcmp(l.name, 'fc8')
        break;
    end
    switch l.type
      case 'conv' % Both conv and fc
          l.learningRate = [0.1, 0.2]; % 1/10 th learning rate
          l.weightDecay = [opts.weightDecay 0];
          l.opts = convOpts;
          l = rmfield(l, 'precious');
          net.layers{end+1} = l;
          prevConv = l.name;
      case 'relu'
          l = rmfield(l, 'leak');
          l = rmfield(l, 'precious');
          net.layers{end+1} = l;
          
          % Add linear mmd loss after relu6 and relu7
          if strcmp(prevConv, 'fc6')
              net = add_lmmdLoss(net, opts, '6') ;
          elseif strcmp(prevConv, 'fc7')
              net = add_lmmdLoss(net, opts, '7') ;
          end
      case 'lrn'
          l = rmfield(l, 'precious');
          net.layers{end+1} = l;
      case 'pool'
          l = rmfield(l, 'precious');
          net.layers{end+1} = l;
    end
end
% Fill in default values
net = vl_simplenn_tidy(net) ; % Update the network to match with the current Matconvnet release
net = add_block(net, opts, '8', 1, 1, 4096, opts.hashSize, 1, 0) ;
net.layers(end) = [] ; % Remove the relu layer that gets added by default to add_block
if opts.batchNormalization, net.layers(end) = [] ; end
