function res = vl_simplenn(net, x, dzdy, res)
% VL_SIMPLENN  Evaluates a simple CNN
%   RES = VL_SIMPLENN(NET, X) evaluates the convnet NET on data X.
%   RES = VL_SIMPLENN(NET, X, DZDY) evaluates the convnent NET and its
%   derivative on data X and output derivative DZDY.
%
%   The network has a simple (linear) topology, i.e. the computational
%   blocks are arranged in a sequence of layers. Please note that
%   there is no need to use this wrapper, which is provided for
%   convenience. Instead, the individual CNN computational blocks can
%   be evaluated directly, making it possible to create significantly
%   more complex topologies, and in general allowing greater
%   flexibility.
%
%   The NET structure contains two fields:
%
%   - net.layers: the CNN layers.
%   - net.normalization: information on how to normalize input data.
%
%   The network expects the data X to be already normalized. This
%   usually involves rescaling the input image(s) and subtracting a
%   mean.
%
%   RES is a structure array with one element per network layer plus
%   one representing the input. So RES(1) refers to the zeroth-layer
%   (input), RES(2) refers to the first layer, etc. Each entry has
%   fields:
%
%   - res(i+1).x: the output of layer i. Hence res(1).x is the network
%     input.
%
%   - res(i+1).aux: auxiliary output data of layer i. For example,
%     dropout uses this field to store the dropout mask.
%
%   - res(i+1).dzdx: the derivative of the network output relative to
%     variable res(i+1).x, i.e. the output of layer i. In particular
%     res(1).dzdx is the derivative of the network output with respect
%     to the network input.
%
%   - res(i+1).dzdw: the derivative of the network output relative to
%     the parameters of layer i. It can be a cell array for multiple
%     parameters.
%
%   net.layers is a cell array of network layers. The following
%   layers, encapsulating corresponding functions in the toolbox, are
%   supported:
%
%   Convolutional layer::
%     The convolutional layer wraps VL_NNCONV(). It has fields:
%
%     - layer.type = 'conv'
%     - layer.filters: the filters.
%     - layer.biases: the biases.
%     - layer.stride: the sampling stride (usually 1).
%     - layer.padding: the padding (usually 0).
%
%   Max pooling layer::
%     The max pooling layer wraps VL_NNPOOL(). It has fields:
%
%     - layer.type = 'pool'
%     - layer.pool: the pooling size.
%     - layer.stride: the sampling stride (usually 1).
%     - layer.padding: the padding (usually 0).
%
%   Normalization layer::
%     The normalization layer wraps VL_NNNORMALIZE(). It has fields
%
%     - layer.type = 'normalize'
%     - layer.param: the normalization parameters.
%
%   ReLU layer::
%     The ReLU layer wraps VL_NNRELU(). It has fields:
%
%     - layer.type = 'relu'
%
%   Dropout layer::
%     The dropout layer wraps VL_NNDROPOUT(). It has fields:
%
%     - layer.type = 'dropout'
%     - layer.rate: the dropout rate.
%
%   Softmax layer::
%     The softmax layer wraps VL_NNSOFTMAX(). It has fields
%
%     - layer.type = 'softmax'
%
%   Log-loss layer::
%     The log-loss layer wraps VL_NNLOSS(). It has fields:
%
%     - layer.type = 'loss'
%     - layer.class: the ground-truth class.
%
%   Softmax-log-loss layer::
%     The softmax-log-loss layer wraps VL_NNSOFTMAXLOSS(). It has
%     fields:
%
%     - layer.type = 'softmaxloss'
%     - layer.class: the ground-truth class.


% Copyright (C) 2014 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

n = numel(net.layers) ;

if (nargin <= 2) || isempty(dzdy)
  doder = false ;
else
  doder = true ;
end

gpuMode = isa(x, 'gpuArray') ;

if (nargin <= 3) || isempty(res)
  res = struct(...
    'x', cell(1,n+1), ...
    'dzdx', cell(1,n+1), ...
    'dzdw', cell(1,n+1), ...
    'aux', cell(1,n+1), ...
    'time', num2cell(zeros(1,n+1)), ...
    'backwardTime', num2cell(zeros(1,n+1))) ;
end
res(1).x = x ;

for i=1:n
  l = net.layers{i} ;
  res(i).time = tic ;
  switch l.type
    case 'conv'
      res(i+1).x = vl_nnconv(res(i).x, l.filters, l.biases, 'pad', l.pad, 'stride', l.stride) ;
    case 'pool'
      res(i+1).x = vl_nnpool(res(i).x, l.pool, 'pad', l.pad, 'stride', l.stride) ;
    case 'normalize'
      res(i+1).x = vl_nnnormalize(res(i).x, l.param) ;
    case 'softmax'
      res(i+1).x = vl_nnsoftmax(res(i).x) ;
    case 'loss'
      res(i+1).x = vl_nnloss(res(i).x, l.class) ;
    case 'softmaxloss'
      res(i+1).x = vl_nnsoftmaxloss(res(i).x, l.class) ;
    case 'relu'
      res(i+1).x = vl_nnrelu(res(i).x) ;
    case 'noffset'
      res(i+1).x = vl_nnnoffset(res(i).x, l.param) ;
    case 'dropout'
      [res(i+1).x, res(i+1).aux] = vl_nndropout(res(i).x, 'rate', l.rate) ;
    otherwise
      error('Unknown layer type %s', l.type);
  end
  try
    %wait(gpuDevice) ;
  catch
    % no gpuDevice
  end
  res(i).time = toc(res(i).time) ;
end

if doder
  res(n+1).dzdx = dzdy ;
  for i=n:-1:1
    l = net.layers{i} ;
    res(i).backwardTime = tic ;
    switch l.type
      case 'conv'
        [res(i).dzdx, res(i).dzdw{1}, res(i).dzdw{2}] = ...
            vl_nnconv(res(i).x, l.filters, l.biases, ...
                      res(i+1).dzdx, ...
                      'pad', l.pad, 'stride', l.stride) ;
      case 'pool'
        res(i).dzdx = vl_nnpool(res(i).x, l.pool, res(i+1).dzdx, ...
          'pad', l.pad, 'stride', l.stride) ;
      case 'normalize'
        res(i).dzdx = vl_nnnormalize(res(i).x, l.param, res(i+1).dzdx) ;
      case 'softmax'
        res(i).dzdx = vl_nnsoftmax(res(i).x, res(i+1).dzdx) ;
      case 'loss'
        res(i).dzdx = vl_nnloss(res(i).x, l.class, res(i+1).dzdx) ;
      case 'softmaxloss'
        res(i).dzdx = vl_nnsoftmaxloss(res(i).x, l.class, res(i+1).dzdx) ;
      case 'relu'
        res(i).dzdx = vl_nnrelu(res(i).x, res(i+1).dzdx) ;
      case 'noffset'
        res(i).dzdx = vl_nnoffset(res(i).x, l.param, res(i+1).dzdx) ;
      case 'dropout'
        res(i).dzdx = vl_nndropout(res(i).x, res(i+1).dzdx, 'mask', res(i+1).aux) ;
    end
    if gpuMode
      %wait(gpuDevice) ;
    end
    res(i).backwardTime = toc(res(i).backwardTime) ;
  end
end
