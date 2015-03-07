function [res, dzdws, arcs] = vl_dagnn(net, inputs, dzdy, res, varargin)
% VL_DAGNN  Evaluates a DAG CNN
%   RES = VL_DAGNN(NET, INPUTS) evaluates the convnet NET on INPUTS.
%   [RES, DZDW] = VL_DAGNN(NET, INPUTS, DZDY) evaluates the convnent NET 
%   and its derivative on INPUTS and output derivative DZDY.
%
%   The network has a direct acyclic topology, i.e. each computational 
%   block can have multiple inputs. Please note that there is no need 
%   to use this wrapper, which is provided for convenience.
%
%   The NET structure must contains a field:
%
%   - net.layers: the CNN layers where each layer is a structure with
%       at least 'type', 'name' and 'inputs' fields.
%
%   RES is a structure array with one element per network layer plus
%   several representing the inputs. So RES(1..numel(INPUTS)) contains the
%   inputs and RES(numel(INPUTS)+1..numel(INPUTS)+numel(NET.layers))
%   contains the layer outputs.
%
%   - res(i+numel(inputs)).x: the output of layer i.
%
%   - res(i+numel(inputs)).aux: auxiliary output data of layer i. 
%     For example, dropout uses this field to store the dropout mask.
%
%   - res(i+numel(inputs)).dzdx: the derivative of the network output 
%     relative to variable res(i+numel(inputs)).x, i.e. the output of 
%     layer i. In particular res(1).dzdx is the derivative of the network 
%     output with respect to the first network input.
%
%   In case of BP step, the output DZDW is a cell array of size 
%   [1, numel(NET.layers)] with derivatives of layer parameters (if any).
%
%   INPUTS is a structure array with two fields:
%
%     - inputs(j).name: The name of the input (string)
%     - inputs(j).x: The input values.
%
%   NET.layers is a cell array of network layers. Each layer must have a 
%   two fields specified:
%
%     - layer.name: layer name (string)
%     - layer.inputs: a cell array of strings which refer either to 
%         another layer name or to some input.
%
%   The following layers, encapsulating corresponding functions in 
%   the toolbox, are supported:
%
%   Convolutional layer (single input)::
%     The convolutional layer wraps VL_NNCONV(). It has fields:
%
%     - layer.type = 'conv'
%     - layer.filters: the filters.
%     - layer.biases: the biases.
%     - layer.stride: the sampling stride (usually 1).
%     - layer.padding: the padding (usually 0).
%
%   Max pooling layer (single input)::
%     The max pooling layer wraps VL_NNPOOL(). It has fields:
%
%     - layer.type = 'pool'
%     - layer.method: pooling method ('max' or 'avg').
%     - layer.pool: the pooling size.
%     - layer.stride: the sampling stride (usually 1).
%     - layer.padding: the padding (usually 0).
%
%   Normalization layer (single input)::
%     The normalization layer wraps VL_NNNORMALIZE(). It has fields
%
%     - layer.type = 'normalize'
%     - layer.param: the normalization parameters.
%
%   ReLU layer (single input)::
%     The ReLU layer wraps VL_NNRELU(). It has fields:
%
%     - layer.type = 'relu'
%
%   Dropout layer (single input)::
%     The dropout layer wraps VL_NNDROPOUT(). It has fields:
%
%     - layer.type = 'dropout'
%     - layer.rate: the dropout rate.
%
%   Softmax layer (single input)::
%     The softmax layer wraps VL_NNSOFTMAX(). It has fields
%
%     - layer.type = 'softmax'
%
%   Log-loss layer (2 inputs)::
%     The log-loss layer wraps VL_NNLOSS(). It has fields:
%
%     - layer.type = 'loss'
%
%     First input are the activations and second input are the ground-truth
%     classes.
%
%   Softmax-log-loss layer (2 input)::
%     The softmax-log-loss layer wraps VL_NNSOFTMAXLOSS(). It has
%     fields:
%
%     - layer.type = 'softmaxloss'
%
%     First input are the activations and second input are the ground-truth
%     classes.
%
%   Concatenation layer (multiple inputs)::
%      Concatenate multiple inputs along selected dimension. It has fields:
%     - layer.type = 'concat'
%     - layer.dim: the dimension along which to concatenate (see help cat)
%
%   Custom layer::
%     This can be used to specify custom layers.
%
%     - layer.type = 'custom'
%     - layer.forward: a function handle computing the block.
%     - layer.backward: a function handle computing the block derivative.
%
%     The first function is called as res(i+1) = forward(layer, res(i), res(i+1))
%     where res() is the struct array specified before. The second function is
%     called as res(i) = backward(layer, res(i), res(i+1)). Note that the
%     `layer` structure can contain additional fields if needed.


% Copyright (C) 2014 Andrea Vedaldi, Karel Lenc
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

% TODO add support for conserve memory

opts.res = [] ;
opts.sync = false ;
opts.disableDropout = false ;
opts = vl_argparse(opts, varargin);

n = numel(net.layers) ;

if (nargin <= 2) || isempty(dzdy)
  doder = false ;
else
  doder = true ;
end

numInputs = numel(inputs);
inputs_x = {inputs.x};

gpuMode = all(cellfun(@(a) isa(a, 'gpuArray'), inputs_x)) ;

[arcs, bufferNames] = vl_dagnn_getarcs(net, inputs);

if nargin <= 3 || isempty(res)
  res = struct(...
    'x', [inputs_x cell(1,n)], ...
    'name', bufferNames, ...
    'dzdx', cell(1,n+numInputs), ...
    'aux', cell(1,n+numInputs), ...
    'time', num2cell(zeros(1,n+numInputs)), ...
    'backwardTime', num2cell(zeros(1,n+numInputs))) ;
else
  for ni = 1:numel(inputs)
    res(ni).x = inputs_x{ni};
  end
end

for li=1:n
  l = net.layers{li} ;
  arcs_sel = arcs(1,:) == li;
  inbi = arcs(2, arcs_sel); % indices of input buffers
  outbi = arcs(3, find(arcs_sel, 1)); % indices of an output buffer
  res(outbi).time = tic ;
  switch l.type
    case 'conv'
      assert(numel(inbi) == 1);
      res(outbi).x = vl_nnconv(res(inbi).x, l.filters, l.biases, ...
        'pad', l.pad, 'stride', l.stride) ;
    case 'pool'
      assert(numel(inbi) == 1);
      res(outbi).x = vl_nnpool(res(inbi).x, l.pool, 'pad', ...
        l.pad, 'stride', l.stride, 'method', l.method) ;
    case 'normalize'
      assert(numel(inbi) == 1);
      res(outbi).x = vl_nnnormalize(res(inbi).x, l.param) ;
    case 'softmax'
      assert(numel(inbi) == 1);
      res(outbi).x = vl_nnsoftmax(res(inbi).x) ;
    case 'loss'
      assert(numel(inbi) == 1);
      res(outbi).x = vl_nnloss(res(inbi).x, l.class) ;
    case 'softmaxloss'
      if(numel(inbi) == 1)
        res(outbi).x = vl_nnsoftmaxloss(res(inbi).x, l.class) ;
      else
        res(outbi).x = vl_nnsoftmaxloss(res(inbi(1)).x, res(inbi(2)).x) ;
      end
    case 'relu'
      assert(numel(inbi) == 1);
      res(outbi).x = vl_nnrelu(res(inbi).x) ;
    case 'noffset'
      assert(numel(inbi) == 1);
      res(outbi).x = vl_nnnoffset(res(inbi).x, l.param) ;
    case 'dropout'
      assert(numel(inbi) == 1);
      if opts.disableDropout
        res(outbi).x = res(inbi).x ;
      else
        [res(outbi).x, res(outbi).aux] = vl_nndropout(res(inbi).x, ...
          'rate', l.rate);
      end
    case 'concat'
      res(outbi).x = vl_nnconcat({res(inbi).x}, l.dim) ;
    case 'custom'
      res(outbi) = l.forward(l, res(inbi), res(bi)) ;
    otherwise
      error('Unknown layer type %s', l.type) ;
  end
  if gpuMode && opts.sync
    % This should make things slower, but on MATLAB 2014a it is necessary
    % for any decent performance.
    wait(gpuDevice) ;
  end
  res(outbi).time = toc(res(outbi).time) ;
end

if doder
  % Initialise to zeros
  for ri = 1:numel(res)-1
    if isempty(res(ri).dzdx)
      res(ri).dzdx = zeros(size(res(ri).x), 'like', res(ri).x);
    else
      res(ri).dzdx(:) = 0;
    end
  end
  res(n + numInputs).dzdx = dzdy ;
  dzdws = cell(1, numel(net.layers)); % Weight derivatives
  
  for li=n:-1:1
    l = net.layers{li} ;
    arcs_sel = arcs(1,:) == li;
    inbi = arcs(2, arcs_sel); % indices of input buffers
    outbi = arcs(3, find(arcs_sel, 1)); % indices of an output buffer
    dzdy = res(outbi).dzdx;
    res(outbi).backwardTime = tic ;
    switch l.type
      case 'conv'
        [dzdx, dzdw_f, dzdw_b] = ...
          vl_nnconv(res(inbi).x, l.filters, l.biases, dzdy, 'pad', ...
          l.pad, 'stride', l.stride) ;
        if isempty(dzdws{li})
          dzdws{li} = {dzdw_f, dzdw_b};
        else
          dzdws{li} = {dzdws{li}{1} + dzdw_f, dzdws{li}{2} + dzdw_b};
        end
      case 'pool'
        dzdx = vl_nnpool(res(inbi).x, l.pool, dzdy, ...
          'pad', l.pad, 'stride', l.stride, 'method', l.method);
      case 'normalize'
        dzdx = vl_nnnormalize(res(inbi).x, l.param, dzdy) ;
      case 'softmax'
        dzdx = vl_nnsoftmax(res(inbi).x, dzdy);
      case 'loss'
        dzdx = vl_nnloss(res(inbi).x, l.class, dzdy);
      case 'softmaxloss'
        dzdx = vl_nnsoftmaxloss(res(inbi).x, l.class, dzdy);
      case 'relu'
        dzdx = vl_nnrelu(res(inbi).x, dzdy);
      case 'noffset'
        dzdx = vl_nnnoffset(res(inbi).x, l.param, dzdy);
      case 'concat'
        dzdx = vl_nnconcat({res(inbi).x}, l.dim, dzdy);
      case 'dropout'
        if opts.disableDropout
          dzdx = dzdy;
        else
          dzdx = vl_nndropout(res(inbi).x, dzdy, 'mask', res(outbi).aux) ;
        end
      case 'custom'
        res(inbi) = l.backward(l, res(inbi), res(outbi)) ;
    end
    % Accummulate the derivatives
    if numel(inbi) > 1
      assert(iscell(dzdx));
      for ini = 1:numel(inbi)
        res(inbi(ini)).dzdx = res(inbi(ini)).dzdx + dzdx{ini};
      end
    else
      res(inbi).dzdx = res(inbi).dzdx + dzdx;
    end
    if gpuMode && opts.sync
      wait(gpuDevice) ;
    end
    res(outbi).backwardTime = toc(res(outbi).backwardTime) ;
  end
end