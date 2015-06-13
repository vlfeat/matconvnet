function net = cnn_imagenet_init(varargin)
% CNN_IMAGENET_INIT  Baseline CNN model

opts.scale = 1 ;
opts.initBias = 0.1 ;
opts.weightDecay = 1 ;
opts.model = 'alexnet' ;
opts = vl_argparse(opts, varargin) ;

% Define layers
switch opts.model
  case 'alexnet'
    net.normalization.imageSize = [227, 227, 3] ;
    net = alexnet(net, opts) ;
  case 'vgg-vd-16'
    net.normalization.imageSize = [224, 224, 3] ;
    net = vgg_vd(net, opts) ;
  case 'vgg-vd-19'
    net.normalization.imageSize = [224, 224, 3] ;
    net = vgg_vd(net, opts) ;
  otherwise
    error('Unknown model ''%s''', opts.model) ;
end
net.layers{end+1} = struct('type', 'softmaxloss', 'name', 'loss') ;

net.normalization.interpolation = 'bicubic' ;
net.normalization.averageImage = [] ;
net.normalization.keepAspect = true ;

% --------------------------------------------------------------------
function net = add_block(net, opts, id, h, w, in, out, stride, pad, init_bias)
% --------------------------------------------------------------------
info = vl_simplenn_display(net) ;
fc = (h == info.dataSize(1,end) && w == info.dataSize(2,end)) ;
if fc
  name = 'fc' ;
else
  name = 'conv' ;
end
net.layers{end+1} = struct('type', 'conv', 'name', sprintf('%s%s', name, id), ...
                           'weights', {{0.01/opts.scale * randn(h, w, in, out, 'single'), ...
                    zeros(out, 1, 'single')}}, ...
                           'stride', stride, ...
                           'pad', pad, ...
                           'learningRate', [1 2], ...
                           'weightDecay', [opts.weightDecay 0]) ;
net.layers{end+1} = struct('type', 'relu', 'name', sprintf('relu%s',id)) ;

% --------------------------------------------------------------------
function net = alexnet(net, opts)
% --------------------------------------------------------------------

net.layers = {} ;

net = add_block(net, opts, '1', 11, 11, 3, 96, 4, 0) ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool1', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'normalize', 'name', 'norm1', ...
                           'param', [5 1 0.0001/5 0.75]) ;

net = add_block(net, opts, '2', 5, 5, 48, 256, 1, 2) ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool2', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'normalize', 'name', 'norm2', ...
                           'param', [5 1 0.0001/5 0.75]) ;

net = add_block(net, opts, '3', 3, 3, 256, 384, 1, 1) ;
net = add_block(net, opts, '4', 3, 3, 192, 384, 1, 1) ;
net = add_block(net, opts, '5', 3, 3, 192, 256, 1, 1) ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool5', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', 0) ;

net = add_block(net, opts, '6', 6, 6, 256, 4096, 1, 0) ;
net.layers{end+1} = struct('type', 'dropout', 'name', 'dropout6', 'rate', 0.5) ;

net = add_block(net, opts, '7', 1, 1, 4096, 4096, 1, 0) ;
net.layers{end+1} = struct('type', 'dropout', 'name', 'dropout7', 'rate', 0.5) ;

net = add_block(net, opts, '8', 1, 1, 4096, 1000, 1, 0) ;
net.layers(end) = [] ;

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

net = add_block(net, opts, '5_1', 3, 3, 256, 512, 1, 1) ;
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
net.layers{end+1} = struct('type', 'dropout', 'name', 'dropout6', 'rate', 0.5) ;

net = add_block(net, opts, '7', 1, 1, 4096, 4096, 1, 0) ;
net.layers{end+1} = struct('type', 'dropout', 'name', 'dropout7', 'rate', 0.5) ;

net = add_block(net, opts, '8', 1, 1, 4096, 1000, 1, 0) ;
net.layers(end) = [] ;

% --------------------------------------------------------------------
function net = vgg_vd_19(net, opts)
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
net.layers{end+1} = struct('type', 'pool', 'name', 'pool3', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0) ;

net = add_block(net, opts, '4_1', 3, 3, 256, 512, 1, 1) ;
net = add_block(net, opts, '4_2', 3, 3, 512, 512, 1, 1) ;
net = add_block(net, opts, '4_3', 3, 3, 512, 512, 1, 1) ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool4', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0) ;

net = add_block(net, opts, '5_1', 3, 3, 256, 512, 1, 1) ;
net = add_block(net, opts, '5_2', 3, 3, 512, 512, 1, 1) ;
net = add_block(net, opts, '5_3', 3, 3, 512, 512, 1, 1) ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool5', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0) ;

net = add_block(net, opts, '6', 7, 7, 512, 4096, 1, 0) ;
net.layers{end+1} = struct('type', 'dropout', 'name', 'dropout6', 'rate', 0.5) ;

net = add_block(net, opts, '7', 1, 1, 4096, 4096, 1, 0) ;
net.layers{end+1} = struct('type', 'dropout', 'name', 'dropout7', 'rate', 0.5) ;

net = add_block(net, opts, '8', 1, 1, 4096, 1000, 1, 0) ;
net.layers(end) = [] ;

