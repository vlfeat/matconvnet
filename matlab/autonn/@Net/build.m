function build(net, varargin)
%BUILD
%   Constructor for a Net. Constructors can't be defined in external file
%   directly, so we use this method.

% Copyright (C) 2016 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).


  % load from struct
  if isscalar(varargin) && isstruct(varargin{1})
    net.loadobj(varargin{1}) ;
    return
  end

  % parse auto-naming option after the other inputs
  opts.sequentialNames = true ;
  [opts, varargin] = vl_argparsepos(opts, varargin) ;

  if isscalar(varargin)
    % a single output layer
    root = varargin{1} ;

    if ~isa(root, 'Layer')  % convert SimpleNN or DagNN to Layer
      root = Layer(root) ;
    end
  else
    % several output layers; create a dummy layer to hold them together
    root = Layer(@cat, 1, varargin{:}) ;
  end

  % make sure all layers have names
  if opts.sequentialNames
    root.sequentialNames() ;
  end

  % figure out the execution order, and list layer objects
  root.resetOrder() ;
  objs = root.buildOrder({}) ;

  % get meta properties from one Layer (ideally they should be merged)
  idx = find(cellfun(@(o) ~isempty(o.meta), objs)) ;
  if ~isempty(idx)
    assert(isscalar(idx), 'More than one Layer has the META property.') ;
    net.meta = objs{idx}.meta ;
  end

  % indexes of callable Layer objects (not Inputs or Params)
  idx = find(cellfun(@(o) ~isa(o, 'Input') && ~isa(o, 'Param'), objs)) ;

  % allocate memory
  net.forward = Net.initStruct(numel(idx), 'func', 'name', ...
      'source', 'args', 'inputVars', 'inputArgPos', 'outputVar') ;
  net.test = net.forward ;
  net.backward = Net.initStruct(numel(idx), 'func', 'name', ...
      'source', 'args', 'inputVars', 'inputArgPos', 'numInputDer') ;

  % there is one var for the output of each Layer in objs; plus another
  % to hold its derivative. note if a Layer has multiple outputs, they
  % can be stored as a nested cell array in the appropriate var.
  net.vars = cell(2 * numel(objs), 1) ;

  numParams = nnz(cellfun(@(o) isa(o, 'Param'), objs)) ;
  net.params = Net.initStruct(numParams, 'name', 'var', ...
      'weightDecay', 'learningRate', 'source', 'trainMethod') ;
  net.inputs = struct() ;

  % first, handle Inputs and Params
  p = 1 ;
  for i = 1:numel(objs)
    obj = objs{i} ;
    if isa(obj, 'Input')
      % an input, store its var index by name
      if isempty(obj.name)  % assign a name automatically
        obj.name = sprintf('input%i', numel(fieldnames(net.inputs)) + 1) ;
      end
      assert(~isfield(net.inputs, obj.name), 'An input with the same name already exists.') ;
      net.inputs.(obj.name) = 2 * i - 1 ;

    elseif isa(obj, 'Param')
      % a learnable parameter, store them in a list
      net.params(p).var = 2 * i - 1 ;
      net.params(p).name = obj.name ;
      net.params(p).weightDecay = obj.weightDecay ;
      net.params(p).learningRate = obj.learningRate ;
      net.params(p).source = obj.source ;

      % store index of training method (defined in Param.trainMethods)
      net.params(p).trainMethod = find(strcmp(obj.trainMethod, Param.trainMethods)) ;

      net.vars{net.params(p).var} = obj.value ;  % set initial value
      p = p + 1 ;
    end
  end

  % store functions for forward pass
  layer = [] ;
  for k = 1:numel(idx)
    obj = objs{idx(k)} ;
    layer.func = obj.func ;
    layer.name = obj.name ;
    layer.source = obj.source ;
    layer.outputVar = 2 * idx(k) - 1 ;
    net.forward(k) = Net.parseArgs(layer, obj.inputs) ;
  end

  % store functions for backward pass
  layer = [] ;
  for k = numel(idx) : -1 : 1
    obj = objs{idx(k)} ;

    % add backward function to execution order
    layer.func = autonn_der(obj.func) ;
    layer.name = obj.name ;
    layer.source = obj.source ;
    layer = Net.parseArgs(layer, obj.inputs) ;

    % figure out position of derivative argument: it's right before
    % the first string (property-value pair), or at the end if none.
    args = layer.args ;
    for lastInput = 0:numel(args)
      if lastInput < numel(args) && ischar(args{lastInput + 1})
        break
      end
    end

    % figure out the number of returned values in bwd mode.
    % assume that the function in bwd mode returns derivatives for
    % all inputs until the last Layer input (e.g. if the 3rd input
    % has class Layer, and others are constants, assume there will be
    % at least 3 output derivatives).
    if isempty(obj.numInputDer)
      layer.numInputDer = max([0, layer.inputArgPos]) ;
    else  % manual override
      layer.numInputDer = obj.numInputDer ;
    end

    % store args for backward mode, with an empty slot for der arg
    layer.args = [args(1:lastInput), {[]}, args(lastInput + 1 : end)] ;

    % modify argument positions according to the new empty slot
    next = layer.inputArgPos > lastInput ;
    layer.inputArgPos(next) = layer.inputArgPos(next) + 1 ;

    % position of der arg
    layer.inputArgPos(end+1) = lastInput + 1 ;

    % its var index: it's the output derivative for the current layer
    layer.inputVars(end+1) = 2 * idx(k) ;

    net.backward(numel(idx) - k + 1) = layer ;
  end

  % store functions for test mode
  layer = [] ;
  for k = 1:numel(idx)
    obj = objs{idx(k)} ;

    % add to execution order
    layer.name = obj.name ;
    layer.source = obj.source ;
    layer.outputVar = 2 * idx(k) - 1 ;

    % default is to use the same arguments
    if isequal(obj.testInputs, 'same')
      args = obj.inputs ;
    else
      args = obj.testInputs ;
    end

    if isempty(obj.testFunc)
      % default is to call the same function as in normal mode
      layer.func = obj.func ;

    elseif isequal(obj.testFunc, 'none')
      % layer is pass-through in test mode (e.g. dropout).
      % we don't fully eliminate the layer in test mode because that
      % would require special handling of in/out var indexes.
      layer.func = @deal ;
      args = args(1) ;  % only deal first input

    else
      % some other function
      layer.func = obj.testFunc ;
    end

    net.test(k) = Net.parseArgs(layer, args) ;
  end

  % network outputs, activate diagnostics automatically if empty
  for k = 1:numel(varargin)
    if isempty(varargin{k}.diagnostics)
      varargin{k}.diagnostics = true ;
    end
  end

  % store diagnostics info for vars
  valid = false(numel(net.vars), 1) ;
  net.diagnostics = Net.initStruct(numel(net.vars), 'var', 'name') ;
  for k = 1 : numel(objs)
    if isequal(objs{k}.diagnostics, true)
      var = objs{k}.outputVar ;
      net.diagnostics(var).var = var ;
      net.diagnostics(var).name = objs{k}.name ;
      net.diagnostics(var + 1).var = var + 1 ;
      net.diagnostics(var + 1).name = ['\partial ', objs{k}.name] ;
      valid([var, var + 1]) = true ;
    end
  end
  net.diagnostics(~valid) = [] ;
end
