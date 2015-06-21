function [res, dzdws] = vl_dagnn(net, inputs, dzdy, res, arcs, varargin)
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

% TODO solve in more elegant way how to store arcs, maybe as a network
% state? Currently it is already too many optional parameteres which need
% to be filled in with empty arrays.

opts.sync = false ;
opts.disableDropout = false ;
opts.forgetRelu = false;
opts.conserveMemory = false;
opts.debugmem = false;
opts = vl_argparse(opts, varargin);

if (nargin <= 2) || isempty(dzdy)
  doder = false ;
else
  doder = true ;
  assert(isfield(dzdy, 'name') && isfield(dzdy, 'dzdx'));
end

if (nargin <= 4) || isempty(arcs)
  arcs = vl_dagnn_getarcs(net, inputs, doder);
end
assert(numel(arcs.bufferNames) == numel(arcs.maps)+numel(inputs));

numArcs = numel(arcs.maps);
numLayers = numel(net.layers) ;
numInputs = numel(inputs);
inputs_x = {inputs.x};
input_isgpu = cellfun(@(a) isa(a, 'gpuArray'), inputs_x);
gpuMode = all(input_isgpu) ;
if ~gpuMode && any(input_isgpu)
  error('Inputs mixed with CPU / GPU arrays'); 
end;

if (nargin <= 3) || isempty(res)
  % Construct a new res structure
  res = struct(...
    'x', [inputs_x cell(1,numLayers)], ...
    'name', arcs.bufferNames, ...
    'dzdx', cell(1,numLayers+numInputs), ...
    'aux', cell(1,numLayers+numInputs), ...
    'time', num2cell(zeros(1,numLayers+numInputs)), ...
    'backwardTime', num2cell(zeros(1,numLayers+numInputs))) ;
else
  assert(numel(res) == numel(arcs.bufferNames));
  % Fill in the inputs
  [res(1:numel(inputs)).x] = deal(inputs_x{1:numel(inputs)}) ;
end

for arci=1:numArcs
  map = arcs.maps(arci);
  l = net.layers{map.layerIdx} ;
  inbis = map.inputIdxs; % indices of input buffers
  outbi = map.outputIdx; % indices of an output buffer
  res(outbi).time = tic ;
  switch l.type
    case 'conv'
      switch numel(inbis)
        case 1
          res(outbi).x = vl_nnconv(res(inbis).x, l.weights{1}, l.weights{2}, ...
            'pad', l.pad, 'stride', l.stride) ;
        case 2
          res(outbi).x = vl_nnconv(res(inbis(1)).x, ...
            res(inbis(2)).x, [], ...
            'pad', l.pad, 'stride', l.stride) ;
        case 3
          res(outbi).x = vl_nnconv(res(inbis(1)).x, ...
            res(inbis(2)).x, res(inbis(3)).x, ...
            'pad', l.pad, 'stride', l.stride) ;
        otherwise
          error('Invalid use of vl_nnconv. Too many inputs');
      end
    case 'times'
      switch numel(inbis)
        case 1
          res(outbi).x = vl_nntimes(res(inbis).x, l.weights{1}) ;
        case 2
          res(outbi).x = vl_nntimes(res(inbis(1)).x, res(inbis(2)).x) ;
        otherwise
          error('Invalid use of vl_nntimes. Too many inputs');
      end
    case 'pool'
      assert(numel(inbis) == 1);
      res(outbi).x = vl_nnpool(res(inbis).x, l.pool, 'pad', ...
        l.pad, 'stride', l.stride, 'method', l.method) ;
    case 'normalize'
      assert(numel(inbis) == 1);
      res(outbi).x = vl_nnnormalize(res(inbis).x, l.param) ;
    case 'spnorm'
      assert(numel(inbis) == 1);
      res(outbi).x = vl_nnspnorm(res(inbis).x, l.param) ;
    case 'softmax'
      assert(numel(inbis) == 1);
      res(outbi).x = vl_nnsoftmax(res(inbis).x) ;
    case 'loss'
      assert(numel(inbis) == 1);
      res(outbi).x = vl_nnloss(res(inbis).x, l.class) ;
    case 'euclidloss'
      if(numel(inbis) == 1)
        if ~isfield(l, 'class')
          error('GT class for softmaxloss not found.'); 
        end;
        res(outbi).x = vl_nneuclidloss(res(inbis).x, l.class) ;
      else
        res(outbi).x = vl_nneuclidloss(res(inbis(1)).x, res(inbis(2)).x) ;
      end
    case 'softmaxloss'
      if(numel(inbis) == 1)
        if ~isfield(l, 'class')
          error('GT class for softmaxloss not found.'); 
        end;
        res(outbi).x = vl_nnsoftmaxloss(res(inbis).x, l.class) ;
      else
        res(outbi).x = vl_nnsoftmaxloss(res(inbis(1)).x, res(inbis(2)).x) ;
      end
    case 'relu'
      assert(numel(inbis) == 1);
      res(outbi).x = vl_nnrelu(res(inbis).x) ;
    case 'noffset'
      assert(numel(inbis) == 1);
      res(outbi).x = vl_nnnoffset(res(inbis).x, l.param) ;
    case 'dropout'
      assert(numel(inbis) == 1);
      if opts.disableDropout
        res(outbi).x = res(inbis).x ;
      else
        [res(outbi).x, res(outbi).aux] = vl_nndropout(res(inbis).x, ...
          'rate', l.rate);
      end
    case 'bnorm'
      res(outbi).x = vl_nnbnorm(res(inbis).x, l.weights{1}, l.weights{2}) ;
    case 'concat'
      res(outbi).x = vl_nnconcat({res(inbis).x}, l.dim) ;
    case 'custom'
      res(outbi) = l.forward(l, res(inbis), res(outbi)) ;
    otherwise
      error('Unknown layer type %s', l.type) ;
  end
  
  % optionally forget intermediate results
  if opts.forgetRelu && strcmp(l.type, 'relu')
    % For RELU -> change the arc for BP and fix the counters
    arcs.maps(arci).inputIdxs = outbi; % keep the dzdxIdxs the same
    if doder
      arcs.bufCounters(inbis) = arcs.bufCounters(inbis) - 1;
      arcs.bufCounters(outbi) = arcs.bufCounters(outbi) + 1;
    end
  end
  
  if gpuMode && opts.sync
    % This should make things slower, but on MATLAB 2014a it is necessary
    % for any decent performance.
    wait(gpuDevice) ;
  end
  % Hack - do not forget losses
  % TODO specify by an argument for the getarcs
  if ~(strcmp(l.type, 'loss') || strcmp(l.type, 'softmaxloss'))
    arcs.bufCounters(inbis) = arcs.bufCounters(inbis) - 1;
  end
  if opts.conserveMemory, res = clearBufs(res, arcs.bufCounters); end;
  res(outbi).time = toc(res(outbi).time) ;
  if opts.debugmem, debugmem(res, {}); end;
end

if doder
  % Fill in the dzdx values
  dzdy_names = {dzdy.name};
  [dzdy_found, dzdyIdxs] = ismember(dzdy_names, arcs.bufferNames);
  if any(~dzdy_found)
    error('DZDY {%s} not found.', ...
      strjoin(dzdy_names(~dzdy_found), ', ')); 
  end;
  dzdy_values = {dzdy.dzdx};
  [res(dzdyIdxs).dzdx] = deal(dzdy_values{:}) ;
  arcs.bufCounters(dzdyIdxs) = arcs.bufCounters(dzdyIdxs) + 1;
  
  dzdws = cell(1, numel(net.layers)); % Weight derivatives
  
  for arci=numArcs:-1:1
    map = arcs.maps(arci);
    li = map.layerIdx;
    l = net.layers{li} ;
    inbis = map.inputIdxs; % indices of input buffers
    outbi = map.outputIdx; % indices of an output buffer
    dzdxis = map.dzdxIdxs;
    assert(numel(inbis) == numel(dzdxis));
    dzdy = res(outbi).dzdx;
    res(outbi).backwardTime = tic ;
    dzdx = [];
    assert(all(arrayfun(@(a) ~isempty(res(a).x), inbis)));
    assert(~isempty(dzdy));
    switch l.type
      case 'conv'
        switch numel(inbis)
          case 1
            dzdw_n = cell(1, 2);
            [dzdx, dzdw_n{1}, dzdw_n{2}] = ...
              vl_nnconv(res(inbis).x, l.weights{1}, l.weights{2}, dzdy, 'pad', ...
              l.pad, 'stride', l.stride) ;
            dzdws{li} = superadd(dzdws{li}, dzdw_n);
          case 2
            [dzdx{1}, dzdx{2}] = ...
              vl_nnconv(res(inbis(1)).x, res(inbis(2)).x, [], ...
              dzdy, 'pad', l.pad, 'stride', l.stride) ;
          case 3
            [dzdx{1}, dzdx{2}, dzdx{3}] = ...
              vl_nnconv(res(inbis(1)).x, res(inbis(2)).x, res(inbis(3)).x, ...
              dzdy, 'pad', l.pad, 'stride', l.stride) ;
          otherwise
            error('Invalid use of vl_nncconv. Too many inputs');
        end
      case 'times'
        switch numel(inbis)
          case 1
            dzdw_n = cell(1, 1);
            [dzdx, dzdw_n{1}] = ...
              vl_nntimes(res(inbis).x, l.weights{1}, dzdy) ;
            dzdws{li} = superadd(dzdws{li}, dzdw_n);
          case 2
            [dzdx{1}, dzdx{2}] = ...
              vl_nntimes(res(inbis(1)).x, res(inbis(2)).x, dzdy) ;
          otherwise
            error('Invalid use of vl_nntimes. Too many inputs');
        end
      case 'bnorm'
        dzdw_n = cell(1, 2);
        [dzdx, dzdw_n{1}, dzdw_n{2}] = ...
          vl_nnbnorm(res(inbis).x, l.weights{1}, l.weights{2}, dzdy) ;
        dzdws{li} = empytadd(dzdws{li}, dzdw_n);
      case 'pool'
        dzdx = vl_nnpool(res(inbis).x, l.pool, dzdy, ...
          'pad', l.pad, 'stride', l.stride, 'method', l.method);
      case 'normalize'
        dzdx = vl_nnnormalize(res(inbis).x, l.param, dzdy) ;
      case 'spnorm'
        dzdx = vl_nnspnorm(res(inbis).x, l.param, dzdy) ;
      case 'softmax'
        dzdx = vl_nnsoftmax(res(inbis).x, dzdy);
      case 'loss'
        dzdx = vl_nnloss(res(inbis).x, l.class, dzdy);
      case 'softmaxloss'
        if(numel(inbis) == 1)
          dzdx = vl_nnsoftmaxloss(res(inbis).x, l.class, dzdy);
        else
          dzdx = cell(1,2);
          dzdx{1} = vl_nnsoftmaxloss(res(inbis(1)).x, res(inbis(2)).x, dzdy);
        end
      case 'euclidloss'
        if(numel(inbis) == 1)
          if ~isfield(l, 'class')
            error('GT class for softmaxloss not found.');
          end;
          dzdx = vl_nneuclidloss(res(inbis).x, l.class, dzdy);
        else
          dzdx = cell(1,2);
          dzdx{1} = vl_nneuclidloss(res(inbis(1)).x, res(inbis(2)).x, dzdy);
        end
      case 'relu'
        dzdx = vl_nnrelu(res(inbis).x, dzdy) ;
      case 'noffset'
        dzdx = vl_nnnoffset(res(inbis).x, l.param, dzdy);
      case 'concat'
        dzdx = vl_nnconcat({res(inbis).x}, l.dim, dzdy);
      case 'dropout'
        if opts.disableDropout
          dzdx = dzdy;
        else
          dzdx = vl_nndropout(res(inbis).x, dzdy, 'mask', res(outbi).aux) ;
        end
      case 'custom'
        % Must perform accummulation! Does not influence the dzdx
        res(dzdxis) = l.backward(l, res(inbis), res(outbi)) ;
      otherwise
        error('Unknown layer');
    end
    % Accummulate the derivatives
    if numel(dzdxis) > 1 && ~isempty(dzdx)
      % Handle multiple-input case
      assert(iscell(dzdx));
      for ini = 1:numel(dzdxis)
        % Do not compute derivatives of inputs (mainly for the loss layers)
        if dzdxis(ini) <= numInputs, continue; end;
        res(dzdxis(ini)).dzdx = superadd(res(dzdxis(ini)).dzdx, dzdx{ini});
      end
    else
      if ~isempty(dzdx)
        res(dzdxis).dzdx = superadd(res(dzdxis).dzdx, dzdx);
      end
    end
    
    if gpuMode && opts.sync
      wait(gpuDevice) ;
    end
    
    arcs.bufCounters(inbis) = arcs.bufCounters(inbis) - 1;
    if opts.conserveMemory
      res = clearBufs(res, arcs.bufCounters);
      res(outbi).dzdx = [];
    end;
    
    res(outbi).backwardTime = toc(res(outbi).backwardTime) ;
    if opts.debugmem, debugmem(res, dzdws); end;
  end
end

function a = superadd(a, b)
% Addition with support for an empty array and cell arrays
if isempty(a)
  a = b;
else
  if iscell(a) && iscell(b)
    assert(numel(a) == numel(b));
    for i = 1:numel(a), a{i} = superadd(a{i}, b{i}); end
  else
    a = a + b;
  end
end

function res = clearBufs(res, bufCounters)
for bfi = find(bufCounters == 0)
  res(bfi).x = [];
end

function debugmem(res, dzdw)
[cpum, gpum] = vl_dagnn_buffersize(res, dzdw);
fprintf('CPU: % 8.2fMB \t GPU: % 8.2fMB\n', cpum./1024^2, gpum./1024^2);