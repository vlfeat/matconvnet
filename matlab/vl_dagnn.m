function res = vl_dagnn(net, x, dzdy, res, varargin)
% VL_DAGNN  Evaluates a DAG CNN
%   RES = VL_DAGNN(NET, X) evaluates the convnet NET on inputs X.
%   RES = VL_DAGNN(NET, X, DZDY) evaluates the convnent NET and its
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
%     - layer.method: pooling method ('max' or 'avg').
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
% TODO solve inputs

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

numInputs = numel(x);
inputNames = {x.name};
inputs = {x.x};

gpuMode = all(cellfun(@(a) isa(a, 'gpuArray'), inputs)) ;

% Check that layer names exist
assert(all(cellfun(@(a) isfield(a, 'name'), net.layers)));
lnames = [inputNames, ...
  cellfun(@(a) a.name, net.layers, 'UniformOutput', false)];

if nargin <= 3 || isempty(res)
  res = struct(...
    'x', [inputs cell(1,n)], ...
    'name', lnames, ...
    'dzdx', cell(1,n+numInputs), ...
    'dzdw', cell(1,n+numInputs), ...
    'aux', cell(1,n+numInputs), ...
    'time', num2cell(zeros(1,n+numInputs)), ...
    'backwardTime', num2cell(zeros(1,n+numInputs))) ;
end

assert(all(cellfun(@(a) isfield(a, 'inputs'), net.layers)));
arcs = getarcs(net.layers, lnames, numInputs);

for li=1:n
  l = net.layers{li} ;
  bi = li + numInputs; % Current buffer index
  res(bi).time = tic ;
  inis = arcs(1, arcs(2,:) == bi); % Input buffer indexes
  inputs = {res(inis).x};
  switch l.type
    case 'conv'
      assert(numel(inputs) == 1);
      res(bi).x = vl_nnconv(inputs{1}, l.filters, l.biases, ...
        'pad', l.pad, 'stride', l.stride) ;
    case 'pool'
      assert(numel(inputs) == 1);
      res(bi).x = vl_nnpool(inputs{1}, l.pool, 'pad', ...
        l.pad, 'stride', l.stride, 'method', l.method) ;
    case 'normalize'
      assert(numel(inputs) == 1);
      res(bi).x = vl_nnnormalize(inputs{1}, l.param) ;
    case 'softmax'
      assert(numel(inputs) == 1);
      res(bi).x = vl_nnsoftmax(inputs{1}) ;
    case 'loss'
      assert(numel(inputs) == 1);
      res(bi).x = vl_nnloss(inputs{1}, l.class) ;
    case 'softmaxloss'
      assert(numel(inputs) == 1);
      res(bi).x = vl_nnsoftmaxloss(inputs{1}, l.class) ;
    case 'relu'
      assert(numel(inputs) == 1);
      res(bi).x = vl_nnrelu(inputs{1}) ;
    case 'noffset'
      assert(numel(inputs) == 1);
      res(bi).x = vl_nnnoffset(inputs{1}, l.param) ;
    case 'dropout'
      assert(numel(inputs) == 1);
      if opts.disableDropout
        res(bi).x = inputs{1} ;
      else
        [res(bi).x, res(bi).aux] = vl_nndropout(inputs{1}, 'rate', l.rate);
      end
    case 'custom'
      res(bi) = l.forward(l, res(inis), res(bi)) ;
    otherwise
      error('Unknown layer type %s', l.type) ;
  end
  if gpuMode && opts.sync
    % This should make things slower, but on MATLAB 2014a it is necessary
    % for any decent performance.
    wait(gpuDevice) ;
  end
  res(bi).time = toc(res(bi).time) ;
end

% TODO check if BP way is correct, maybe with testing whether a chain
% network works?

if doder
  res(n + numInputs).dzdx = dzdy ;
  
  for li=n:-1:1
    l = net.layers{li} ;
    % The last already has the dzdx set
    bi = li + numInputs - 1;
    res(bi).backwardTime = tic ;
    succsIdxs = arcs(2, arcs(1,:) == bi);
    dzdxs = {res(succsIdxs).dzdx};
    switch l.type
      case 'conv'
        [res(bi).dzdx, res(bi).dzdw{1}, res(bi).dzdw{2}] = funsum(...
          @(dzdx) vl_nnconv(res(bi).x, l.filters, l.biases, ...
            dzdx, 'pad', l.pad, 'stride', l.stride), ...
          dzdxs, 3) ;
      case 'pool'
        res(bi).dzdx = funsum(...
          @(dzdx) vl_nnpool(res(bi).x, l.pool, dzdx, ...
            'pad', l.pad, 'stride', l.stride, 'method', l.method),...
          dzdxs, 1) ;
      case 'normalize'
        res(bi).dzdx = funsum(...
          @(dzdx) vl_nnnormalize(res(bi).x, l.param, dzdx), ...
          dzdxs, 1) ;
      case 'softmax'
        res(bi).dzdx = funsum(...
          @(dzdx) vl_nnsoftmax(res(bi).x, dzdx), ...
          dzdxs, 1);
      case 'loss'
        res(bi).dzdx = funsum(...
          @(dzdx) vl_nnloss(res(bi).x, l.class, dzdx), ...
          dzdxs, 1);
      case 'softmaxloss'
        res(bi).dzdx = funsum(...
          @(dzdx) vl_nnsoftmaxloss(res(bi).x, l.class, dzdx), ...
          dzdxs, 1);
      case 'relu'
        res(bi).dzdx = funsum(...
          @(dzdx) vl_nnrelu(res(bi).x, dzdx), ...
          dzdxs, 1);
      case 'noffset'
        res(bi).dzdx = funsum(...
          @(dzdx) vl_nnnoffset(res(bi).x, l.param, dzdx), ...
          dzdxs, 1);
      case 'dropout'
        if opts.disableDropout
          res(bi).dzdx = funsum(@(dzdx) dzdx, dzdxs, 1) ;
        else
          res(bi).dzdx = funsum(...
            @(dzdx) vl_nndropout(res(bi).x, dzdx, 'mask', res(bi).aux), ...
            dzdxs, 1) ;
        end
      case 'custom'
        res(bi) = l.backward(l, res(bi), res(succsIdxs)) ;
    end
    if gpuMode && opts.sync
      wait(gpuDevice) ;
    end
    res(bi).backwardTime = toc(res(bi).backwardTime) ;
  end
end


function arcs = getarcs(layers, lnames, numInputs)
arcs = [];
for li = 1:numel(layers)
  inputs = layers{li}.inputs;
  [tfnd, pred] = ismember(inputs, lnames);
  if any(tfnd == 0)
    error('Inputs {%s} for layer %s not found', ...
      strjoin(inputs(~tfnd), ', '), layers{li}.name);
  end;
  arcs = [arcs [pred; (li+numInputs)*ones(1, numel(pred))]];
end

function varargout = funsum(fun, args, numArgOut)
% FUNSUM Accummulate function results
%   [OUT1, OUT2, ...] = FUNSUM(FUN, ARGS, NUMARGS) Calls function FUN for
%   each argument from cell array ARGS and accummulate NUMARGS outputs.
%   Currently supports at most 3 output arguments.

assert(numel(args) >= 1);
assert(numArgOut >= 1 && numArgOut <= 3);

varargout = funcal(fun, args{1}, numArgOut);
for ai = 2:numel(args)
  % Unfortunately, matlab does not have in-place sum
  tmp = funcal(fund, args{ai}, numArgOut);
  varargout = arrayfun(@(i) varargout{i} + tmp{i}, 'UniformOutput', false);
end

function out = funcal(fun, arg, numArgOut)
switch numArgOut
  case 1
    [out1] = fun(arg);
    out = {out1};
  case 2
    [out1, out2] = fun(arg);
    out = {out1, out2};
  case 3
    [out1, out2, out3] = fun(arg);
    out = {out1, out2, out3};
end