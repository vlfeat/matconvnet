function [net, loss, top1err, top5err] = cnn_imagenet_init_autonn(varargin)
% CNN_IMAGENET_INIT_AUTONN  Initialize a standard CNN for ImageNet

opts.scale = 1 ;
opts.initBias = 0.1 ;
opts.weightDecay = 1 ;
%opts.weightInitMethod = 'xavierimproved' ;
opts.weightInitMethod = 'gaussian' ;
opts.model = 'alexnet' ;
opts.batchNormalization = false ;
opts.networkType = 'autonn' ;
opts.cudnnWorkspaceLimit = 1024*1024*1204 ; % 1GB
opts.normalization = [5 1 0.0001/5 0.75] ; % for vl_nnnormalize layer
opts = vl_argparse(opts, varargin) ;

assert(strcmp(opts.networkType, 'autonn')) ;

% Define layers
imageSize = [224, 224, 3] ;
switch opts.model
  case 'alexnet'
    imageSize = [227, 227, 3] ;
    prediction = alexnet(opts) ;
    bs = 256 ;
  case 'vgg-f'
    prediction = vgg_f(opts) ;
    bs = 256 ;
  case 'vgg-m'
    prediction = vgg_m(opts) ;
    bs = 196 ;
  case 'vgg-s'
    prediction = vgg_s(opts) ;
    bs = 128 ;
  case 'vgg-vd-16'
    prediction = vgg_vd(opts) ;
%     bs = 32 ;
    bs = 24 ;
  case 'vgg-vd-19'
    prediction = vgg_vd(opts) ;
    bs = 24 ;
  otherwise
    error('Unknown model ''%s''', opts.model) ;
end

% % final touches
% switch lower(opts.weightInitMethod)
%   case {'xavier', 'xavierimproved'}
%     net.layers{end}.weights{1} = net.layers{end}.weights{1} / 10 ;
% end

% create loss and error metrics
label = Input('label') ;
loss = vl_nnsoftmaxloss(prediction, label) ;
top1err = vl_nnloss(prediction, label, 'loss', 'classerror') ;
top5err = vl_nnloss(prediction, label, 'loss', 'topkerror', 'topK', 5) ;

% assign names automatically, and compile network
Layer.workspaceNames() ;
net = Net(loss, top1err, top5err) ;

% Meta parameters
net.meta.normalization.imageSize = imageSize ;
net.meta.inputSize = net.meta.normalization.imageSize ;
net.meta.normalization.border = 256 - net.meta.normalization.imageSize(1:2) ;
net.meta.normalization.interpolation = 'bicubic' ;
net.meta.normalization.averageImage = [] ;
net.meta.normalization.keepAspect = true ;
net.meta.augmentation.rgbVariance = zeros(0,3) ;
net.meta.augmentation.transformation = 'stretch' ;

if ~opts.batchNormalization
  lr = logspace(-2, -4, 60) ;
else
  lr = logspace(-1, -4, 20) ;
end

net.meta.trainOpts.learningRate = lr ;
net.meta.trainOpts.numEpochs = numel(lr) ;
net.meta.trainOpts.batchSize = bs ;
net.meta.trainOpts.weightDecay = 0.0005 ;

% --------------------------------------------------------------------
function net = add_block(net, opts, sz, varargin)
% --------------------------------------------------------------------
filters = Param('value', init_weight(opts, sz, 'single'), 'learningRate', 1) ;
biases = Param('value', zeros(sz(4), 1, 'single'), 'learningRate', 2) ;

net = vl_nnconv(net, filters, biases, varargin{:}, ...
    'CudnnWorkspaceLimit', opts.cudnnWorkspaceLimit) ;

if opts.batchNormalization
  net = vl_nnbnorm(net, 'learningRate', [2 1 0.05]) ;
end

net = vl_nnrelu(net) ;

% -------------------------------------------------------------------------
function weights = init_weight(opts, sz, type)
% -------------------------------------------------------------------------
% See K. He, X. Zhang, S. Ren, and J. Sun. Delving deep into
% rectifiers: Surpassing human-level performance on imagenet
% classification. CoRR, (arXiv:1502.01852v1), 2015.

switch lower(opts.weightInitMethod)
  case 'gaussian'
    sc = 0.01/opts.scale ;
    weights = randn(sz, type)*sc;
  case 'xavier'
    sc = sqrt(3/(h*w*in)) ;
    weights = (rand(sz, type)*2 - 1)*sc ;
  case 'xavierimproved'
    sc = sqrt(2/(h*w*out)) ;
    weights = randn(sz, type)*sc ;
  otherwise
    error('Unknown weight initialization method''%s''', opts.weightInitMethod) ;
end

% --------------------------------------------------------------------
function net = alexnet(opts)
% --------------------------------------------------------------------
net = Input('input') ;
bn = opts.batchNormalization ;

net = add_block(net, opts, [11, 11, 3, 96], 'stride', 4, 'pad', 0) ;
if ~bn
  net = vl_nnnormalize(net, opts.normalization) ;
end
net = vl_nnpool(net, [3 3], 'stride', 2, 'pad', 0) ;

net = add_block(net, opts, [5, 5, 48, 256], 'stride', 1, 'pad', 2) ;
if ~bn
  net = vl_nnnormalize(net, opts.normalization) ;
end
net = vl_nnpool(net, [3 3], 'stride', 2, 'pad', 0) ;

net = add_block(net, opts, [3, 3, 256, 384], 'stride', 1, 'pad', 1) ;
net = add_block(net, opts, [3, 3, 192, 384], 'stride', 1, 'pad', 1) ;
net = add_block(net, opts, [3, 3, 192, 256], 'stride', 1, 'pad', 1) ;
net = vl_nnpool(net, [3 3], 'stride', 2, 'pad', 0) ;

net = add_block(net, opts, [6, 6, 256, 4096], 'stride', 1, 'pad', 0) ;
if ~bn
  net = vl_nndropout(net) ;
end

net = add_block(net, opts, [1, 1, 4096, 4096], 'stride', 1, 'pad', 0) ;
if ~bn
  net = vl_nndropout(net) ;
end

net = add_block(net, opts, [1, 1, 4096, 1000], 'stride', 1, 'pad', 0) ;

net = net.inputs{1} ;  % delete last layer
if bn, net = net.inputs{1} ; end

% --------------------------------------------------------------------
function net = vgg_s(opts)
% --------------------------------------------------------------------
net = Input('input') ;
bn = opts.batchNormalization ;

net = add_block(net, opts, [7, 7, 3, 96], 'stride', 2, 'pad', 0) ;
if ~bn
  net = vl_nnnormalize(net, opts.normalization) ;
end
net = vl_nnpool(net, [3 3], 'stride', 3, 'pad', [0 2 0 2]) ;

net = add_block(net, opts, [5, 5, 96, 256], 'stride', 1, 'pad', 0) ;
if ~bn
  net = vl_nnnormalize(net, opts.normalization) ;
end
net = vl_nnpool(net, [2 2], 'stride', 2, 'pad', [0 1 0 1]) ;

net = add_block(net, opts, [3, 3, 256, 512], 'stride', 1, 'pad', 1) ;
net = add_block(net, opts, [3, 3, 512, 512], 'stride', 1, 'pad', 1) ;
net = add_block(net, opts, [3, 3, 512, 512], 'stride', 1, 'pad', 1) ;
net = vl_nnpool(net, [3 3], 'stride', 3, 'pad', [0 1 0 1]) ;

net = add_block(net, opts, [6, 6, 512, 4096], 'stride', 1, 'pad', 0) ;
if ~bn
  net = vl_nndropout(net) ;
end

net = add_block(net, opts, [1, 1, 4096, 4096], 'stride', 1, 'pad', 0) ;
if ~bn
  net = vl_nndropout(net) ;
end

net = add_block(net, opts, [1, 1, 4096, 1000], 'stride', 1, 'pad', 0) ;

net = net.inputs{1} ;  % delete last layer
if bn, net = net.inputs{1} ; end

% --------------------------------------------------------------------
function net = vgg_m(opts)
% --------------------------------------------------------------------
net = Input('input') ;
bn = opts.batchNormalization ;

net = add_block(net, opts, [7, 7, 3, 96], 'stride', 2, 'pad', 0) ;
if ~bn
  net = vl_nnnormalize(net, opts.normalization) ;
end
net = vl_nnpool(net, [3 3], 'stride', 2, 'pad', 0) ;

net = add_block(net, opts, [5, 5, 96, 256], 'stride', 2, 'pad', 1) ;
if ~bn
  net = vl_nnnormalize(net, opts.normalization) ;
end
net = vl_nnpool(net, [3 3], 'stride', 2, 'pad', [0 1 0 1]) ;

net = add_block(net, opts, [3, 3, 256, 512], 'stride', 1, 'pad', 1) ;
net = add_block(net, opts, [3, 3, 512, 512], 'stride', 1, 'pad', 1) ;
net = add_block(net, opts, [3, 3, 512, 512], 'stride', 1, 'pad', 1) ;
net = vl_nnpool(net, [3 3], 'stride', 2, 'pad', 0) ;

net = add_block(net, opts, [6, 6, 512, 4096], 'stride', 1, 'pad', 0) ;
if ~bn
  net = vl_nndropout(net) ;
end

net = add_block(net, opts, [1, 1, 4096, 4096], 'stride', 1, 'pad', 0) ;
if ~bn
  net = vl_nndropout(net) ;
end

net = add_block(net, opts, [1, 1, 4096, 1000], 'stride', 1, 'pad', 0) ;

net = net.inputs{1} ;  % delete last layer
if bn, net = net.inputs{1} ; end

% --------------------------------------------------------------------
function net = vgg_f(opts)
% --------------------------------------------------------------------
net = Input('input') ;
bn = opts.batchNormalization ;

net = add_block(net, opts, [11, 11, 3, 64], 'stride', 4, 'pad', 0) ;
if ~bn
  net = vl_nnnormalize(net, opts.normalization) ;
end
net = vl_nnpool(net, [3 3], 'stride', 2, 'pad', [0 1 0 1]) ;

net = add_block(net, opts, [5, 5, 64, 256], 'stride', 1, 'pad', 2) ;
if ~bn
  net = vl_nnnormalize(net, opts.normalization) ;
end
net = vl_nnpool(net, [3 3], 'stride', 2, 'pad', 0) ;

net = add_block(net, opts, [3, 3, 256, 256], 'stride', 1, 'pad', 1) ;
net = add_block(net, opts, [3, 3, 256, 256], 'stride', 1, 'pad', 1) ;
net = add_block(net, opts, [3, 3, 256, 256], 'stride', 1, 'pad', 1) ;
net = vl_nnpool(net, [3 3], 'stride', 2, 'pad', 0) ;

net = add_block(net, opts, [6, 6, 256, 4096], 'stride', 1, 'pad', 0) ;
if ~bn
  net = vl_nndropout(net) ;
end

net = add_block(net, opts, [1, 1, 4096, 4096], 'stride', 1, 'pad', 0) ;
if ~bn
  net = vl_nndropout(net) ;
end

net = add_block(net, opts, [1, 1, 4096, 1000], 'stride', 1, 'pad', 0) ;

net = net.inputs{1} ;  % delete last layer
if bn, net = net.inputs{1} ; end

% --------------------------------------------------------------------
function net = vgg_vd(opts)
% --------------------------------------------------------------------
net = Input('input') ;
bn = opts.batchNormalization ;

net = add_block(net, opts, [3, 3, 3, 64], 'stride', 1, 'pad', 1) ;
net = add_block(net, opts, [3, 3, 64, 64], 'stride', 1, 'pad', 1) ;
net = vl_nnpool(net, [2 2], 'stride', 2, 'pad', 0) ;

net = add_block(net, opts, [3, 3, 64, 128], 'stride', 1, 'pad', 1) ;
net = add_block(net, opts, [3, 3, 128, 128], 'stride', 1, 'pad', 1) ;
net = vl_nnpool(net, [2 2], 'stride', 2, 'pad', 0) ;

net = add_block(net, opts, [3, 3, 128, 256], 'stride', 1, 'pad', 1) ;
net = add_block(net, opts, [3, 3, 256, 256], 'stride', 1, 'pad', 1) ;
net = add_block(net, opts, [3, 3, 256, 256], 'stride', 1, 'pad', 1) ;
if strcmp(opts.model, 'vgg-vd-19')
  net = add_block(net, opts, [3, 3, 256, 256], 'stride', 1, 'pad', 1) ;
end
net = vl_nnpool(net, [2 2], 'stride', 2, 'pad', 0) ;

net = add_block(net, opts, [3, 3, 256, 512], 'stride', 1, 'pad', 1) ;
net = add_block(net, opts, [3, 3, 512, 512], 'stride', 1, 'pad', 1) ;
net = add_block(net, opts, [3, 3, 512, 512], 'stride', 1, 'pad', 1) ;
if strcmp(opts.model, 'vgg-vd-19')
  net = add_block(net, opts, [3, 3, 512, 512], 'stride', 1, 'pad', 1) ;
end
net = vl_nnpool(net, [2 2], 'stride', 2, 'pad', 0) ;

net = add_block(net, opts, [3, 3, 512, 512], 'stride', 1, 'pad', 1) ;
net = add_block(net, opts, [3, 3, 512, 512], 'stride', 1, 'pad', 1) ;
net = add_block(net, opts, [3, 3, 512, 512], 'stride', 1, 'pad', 1) ;
if strcmp(opts.model, 'vgg-vd-19')
  net = add_block(net, opts, [3, 3, 512, 512], 'stride', 1, 'pad', 1) ;
end
net = vl_nnpool(net, [2 2], 'stride', 2, 'pad', 0) ;

net = add_block(net, opts, [7, 7, 512, 4096], 'stride', 1, 'pad', 0) ;
if ~bn
  net = vl_nndropout(net) ;
end

net = add_block(net, opts, [1, 1, 4096, 4096], 'stride', 1, 'pad', 0) ;
if ~bn
  net = vl_nndropout(net) ;
end

net = add_block(net, opts, [1, 1, 4096, 1000], 'stride', 1, 'pad', 0) ;

net = net.inputs{1} ;  % delete last layer
if bn, net = net.inputs{1} ; end
