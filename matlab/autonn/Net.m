classdef Net < handle
  properties (SetAccess = protected, GetAccess = public)
    forward = []  % forward pass function calls
    backward = []  % backward pass function calls
    test = []  % test mode function calls
    vars = {}  % cell array of variables and their derivatives
    inputs = []  % struct of network's Inputs, indexed by name
    params = []  % list of Params
  end

  methods
    function net = Net(root)
      % figure out the execution order, and list layer objects
      root.resetOrder() ;
      objs = root.buildOrder({}) ;
      
      % indexes of callable layer objects (not Inputs or Params)
      idx = find(cellfun(@(o) ~isa(o, 'Input') && ~isa(o, 'Param'), objs)) ;
      
      % allocate memory
      net.forward = Net.initStruct(numel(idx), 'func', 'name', ...
          'args', 'inputVars', 'inputArgPos', 'outputVar') ;
      net.test = net.forward ;
      net.backward = Net.initStruct(numel(idx), 'func', 'name', ...
          'args', 'inputVars', 'inputArgPos', 'numInputDer') ;
      
      net.vars = cell(2 * numel(objs), 1) ;
      
      numParams = nnz(cellfun(@(o) isa(o, 'Param'), objs)) ;
      net.params = Net.initStruct(numParams, 'name', 'idx', 'weightDecay', 'learningRate') ;
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
          net.inputs.(obj.name) = i ;
          
        elseif isa(obj, 'Param')
          % a learnable parameter, store them in a list
          net.params(p).idx = i ;
          net.params(p).name = obj.name ;
          net.params(p).weightDecay = obj.weightDecay ;
          net.params(p).learningRate = obj.learningRate ;
          net.vars{2 * i - 1} = obj.value ;  % set initial value
          p = p + 1 ;
        end
      end
      
      % store functions for forward pass
      layer = [] ;
      for k = 1:numel(idx)
        obj = objs{idx(k)} ;
        layer.func = obj.func ;
        layer.name = obj.name ;
        layer.outputVar = 2 * idx(k) - 1 ;
        net.forward(k) = Net.parseArgs(layer, obj.inputs) ;
      end
      
      % store functions for backward pass
      layer = [] ;
      for k = numel(idx) : -1 : 1
        obj = objs{idx(k)} ;
        
        % add backward function to execution order
        layer.func = der(obj.func) ;
        layer.name = obj.name ;
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
          args = args{1} ;  % only deal first input
          
        else
          % some other function
          layer.func = obj.testFunc ;
        end
        
        net.test(k) = Net.parseArgs(layer, args) ;
      end
    end
    
    function move(net, device)
    %MOVE Move data to CPU or GPU
    %  MOVE(DESTINATION) moves the data associated to the net object OBJ
    %  to either the 'gpu' or the 'cpu'.
      switch device
        case 'gpu', moveOp = @gpuArray ;
        case 'cpu', moveOp = @gather ;
        otherwise, error('Unknown device ''%s''.', device) ;
      end
      
      net.vars = cellfun(moveOp, net.vars, 'UniformOutput',false) ;
    end
    
    function eval(net, mode, derOutput, accumulateParamDers)
      if nargin < 2
        mode = 'normal' ;
      end
      if nargin < 3
        derOutput = 1 ;
      end
      if nargin < 4
        accumulateParamDers = false ;
      end
      
      % use local variables for efficiency
      vars = net.vars ;
      net.vars = {} ;  % allows Matlab to release memory when needed
      
      switch mode
      case {'normal', 'forward'}  % forward and backward
        forward = net.forward ;   %#ok<*PROPLC> % disable MLint's complaints
      case 'test'  % test mode
        forward = net.test ;
      otherwise
        error('Unknown mode ''%s''.', mode) ;
      end
      
      % forward pass
      for k = 1:numel(forward)
        layer = forward(k) ;
        args = layer.args ;
        args(layer.inputArgPos) = vars(layer.inputVars) ;
        vars{layer.outputVar} = layer.func(args{:}) ;
      end
      
      % backward pass
      if strcmp(mode, 'normal')
        % clear all derivatives. derivatives are even-numbered vars.
        clear = repmat([false; true], numel(vars) / 2, 1);
        if accumulateParamDers  % except for params (e.g. to implement sub-batches)
          clear([net.params.idx] * 2) = false ;
        end
        [vars(clear)] = deal({0}) ;
        
        % set root layer's output derivative
        assert(~isempty(derOutput), 'Must specify non-empty output derivatives for normal mode.')
        vars{end} = derOutput ;
        
        backward = net.backward ;
        
        for k = 1:numel(backward)
          % populate function arguments with input vars and derivatives
          layer = backward(k) ;
          args = layer.args ;
          inputArgPos = layer.inputArgPos ;
          args(inputArgPos) = vars(layer.inputVars) ;
          
          % call function and collect outputs
          out = cell(1, layer.numInputDer) ;
          [out{:}] = layer.func(args{:}) ;
          
          % sum derivatives. the derivative var corresponding to each input
          % comes right next to it in the vars list. note that some outputs
          % may be ignored (because they're not input layers, just constant
          % arguments).
          inputDers = layer.inputVars(1:end-1) + 1 ;  % last input is dzdy, doesn't count
          for i = find(inputArgPos <= numel(out))
            vars{inputDers(i)} = vars{inputDers(i)} + out{inputArgPos(i)} ;
          end
        end
      end
      
      net.vars = vars ;
    end
    
    function value = getValue(net, layer)
      if ~isnumeric(layer)
        assert(isa(layer, 'Layer'), 'LAYER must either be layer indexes or a Layer object.') ;
        layer = layer.idx ;
      end
      if isscalar(layer)
        value = net.vars{2 * layer - 1} ;
      else
        value = net.vars(2 * layer - 1) ;
      end
    end
    
    
    function der = getDer(net, layer)
      if ~isnumeric(layer)
        assert(isa(layer, 'Layer'), 'LAYER must either be layer indexes or a Layer object.') ;
        layer = layer.idx ;
      end
      if isscalar(layer)
        der = net.vars{2 * layer} ;
      else
        der = net.vars(2 * layer) ;
      end
    end
    
    function setValue(net, layer, value)
      if ~isnumeric(layer)
        assert(isa(layer, 'Layer'), 'LAYER must either be layer indexes or a Layer object.') ;
        layer = layer.idx ;
      end
      if isscalar(layer)
        net.vars{2 * layer - 1} = value ;
      else
        net.vars(2 * layer - 1) = value ;
      end
    end
    
    function setDer(net, layer, der)
      if ~isnumeric(layer)
        assert(isa(layer, 'Layer'), 'LAYER must either be layer indexes or a Layer object.') ;
        layer = layer.idx ;
      end
      if isscalar(layer)
        net.vars{2 * layer} = der ;
      else
        net.vars(2 * layer) = der ;
      end
    end
    
    function setInputs(net, varargin)
      for i = 1 : 2 : numel(varargin) - 1
        net.vars{2 * net.inputs.(varargin{i}) - 1} = varargin{i+1} ;
      end
    end
    
    function displayVars(net)
      % get information on each var, and corresponding derivative
      n = numel(net.vars) / 2 ;
      assert(n ~= 0, 'NET.VARS is empty.') ;
      type = cell(n, 1) ;
      names = cell(n, 1) ;
      funcs = cell(n, 1) ;
      
      % vars that correspond to inputs
      inputNames = fieldnames(net.inputs);
      for k = 1:numel(inputNames)
        idx = net.inputs.(inputNames{k}) ;
        type{idx} = 'Input' ;
        names{idx} = inputNames{k} ;
      end
      
      % vars that correspond to params
      [type{[net.params.idx]}] = deal('Param') ;
      names([net.params.idx]) = {net.params.name} ;
      
      % vars that correspond to layer outputs
      idx = ([net.forward.outputVar] + 1) / 2 ;
      [type{idx}] = deal('Layer') ;
      names(idx) = {net.forward.name} ;
      funcs(idx) = cellfun(@func2str, {net.forward.func}, 'UniformOutput', false) ;
      
      % get Matlab to print the values nicely (e.g. "[50x1 double]")
      values = strsplit(evalc('disp(net.vars)'), '\n') ;
      values = cellfun(@strtrim, values, 'UniformOutput', false) ;
      
      % now print out the info as a table
      str = repmat(' ', n + 1, 1) ;
      str = [str, char('Type', type{:})] ;
      str(:,end+1:end+2) = ' ' ;
      str = [str, char('Function', funcs{:})] ;
      str(:,end+1:end+2) = ' ' ;
      str = [str, char('Name', names{:})] ;
      str(:,end+1:end+2) = ' ' ;
      str = [str, char('Value', values{1 : 2 : 2 * n})] ;
      str(:,end+1:end+2) = ' ' ;
      str = [str, char('Derivative', values{2 : 2 : 2 * n})] ;
      
      disp(str) ;
    end
    
    function display(net, name)
      if nargin < 2
        name = inputname(1) ;
      end
      fprintf('\n%s = \n', name) ;
      disp(net) ;
      
      if ~isempty(name)
        fprintf('  <a href="matlab:%s.displayVars()">Display variables</a>\n\n', name) ;
      end
    end
    
    function s = saveobj(net)
      s.forward = net.forward ;
      s.backward = net.backward ;
      s.test = net.test ;
      s.vars = net.vars ;
      s.inputs = net.inputs ;
      s.params = net.params ;
    end
  end
  
  methods (Static)
    function net = loadobj(s)
      net.forward = s.forward ;
      net.backward = s.backward ;
      net.test = s.test ;
      net.vars = s.vars ;
      net.inputs = s.inputs ;
      net.params = s.params ;
    end
  end
  
  methods (Static, Access = private)
    function layer = parseArgs(layer, args)
      % helper function to parse a layer's arguments, storing the constant
      % arguments (args), non-constant var indexes (inputVars), and their
      % positions in the arguments list (inputArgPos).
      inputLayers = [] ;
      inputArgPos = [] ;
      for a = 1:numel(args)
        if isa(args{a}, 'Layer')
          inputLayers(end+1) = args{a}.idx ;  %#ok<*AGROW>
          inputArgPos(end+1) = a ;
          args{a} = [] ;
        end
      end
      layer.args = args ;
      layer.inputVars = 2 * inputLayers - 1 ;
      layer.inputArgPos = inputArgPos ;
      layer = orderfields(layer) ;  % have a consistent field order, to not botch assignments
    end
    
    function s = initStruct(n, varargin)
      % helper function to initialize a struct with given fields and size.
      % note fields are sorted in ASCII order (important when assigning
      % structs).
      varargin(2,:) = {cell(1, n)} ;
      s = orderfields(struct(varargin{:})) ;
    end
  end
end

