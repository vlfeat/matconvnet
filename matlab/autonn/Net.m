classdef Net < handle
  properties
    conserveMemory = true
  end
  
  properties %(Access = protected)
    layers = []
    inputs = []  % struct of network's Inputs, indexed by name
    params = []  % list of Params
    
    nonParams = []  % complement of the list of Params
    order = []
  end

  methods
    function net = Net(root)
      % figure out the forward execution order, and list layer objects
      root.resetOrder() ;
      objs = root.buildOrder({}) ;
      
      % store function call info of layer objects in net.layers struct
      net.layers(numel(objs)).value = [] ;  % output value
      net.layers(numel(objs)).der = [] ;  % output derivative
      net.inputs = struct() ;
      
      for k = 1:numel(objs)
        % store function handle and name
        net.layers(k).func = objs{k}.func ;
        net.layers(k).name = objs{k}.name ;
        
        if isa(objs{k}, 'Input')
          % an input, allow it to be searched by name
          if isempty(objs{k}.name)  % assign a name automatically
            objs{k}.name = sprintf('input%i', numel(fieldnames(net.inputs)) + 1) ;
          end
          assert(~isfield(net.inputs, objs{k}.name), 'An input with the same name already exists.') ;
          net.inputs.(objs{k}.name) = k ;
          
        elseif isa(objs{k}, 'Param')
          % a learnable parameter, store them sequentially
          net.params(end+1).idx = k ;
          net.params(end).weightDecay = objs{k}.weightDecay ;
          net.params(end).learningRate = objs{k}.learningRate ;
          net.layers(k).value = objs{k}.value ;  % initial value
          
        else
          % store input layer indexes and their position in arguments list
          net.layers(k).inputLayerIdx = [] ;
          net.layers(k).inputArgPos = [] ;
          net.layers(k).inputIsConst = [] ;
          args = objs{k}.inputs ;
          inputHasDer = false(0, 1) ;
          
          for a = 1:numel(args)
            if isa(args{a}, 'Layer')
              net.layers(k).inputLayerIdx(end+1) = args{a}.idx ;
              net.layers(k).inputArgPos(end+1) = a ;
              inputHasDer(end+1) = isa(args{a}, 'Param') || ...
                ~isempty(net.layers(args{a}.idx).inputLayerIdx) ;  %#ok<AGROW>
              args{a} = [] ;
            end
          end
          
          % store remaining arguments (constants)
          net.layers(k).args = args ;
          
          % figure out position of derivative argument: it's right before the
          % first string (property-value pair), or at the end if there's none.
          for a = 0:numel(args)
            if a < numel(args) && ischar(args{a + 1})
              break
            end
          end
          net.layers(k).derArgPos = a + 1 ;
          
          % store args for backward mode, with an empty slot for der arg
          net.layers(k).bwdArgs = [args(1:a), {[]}, args(a + 1 : end)] ;
          
          % figure out the number of returned values in bwd mode.
          % assume that the function in bwd mode returns derivatives for all
          % inputs until the last Layer input (e.g. if the 3rd input has
          % class Layer, and others are constants, assume there will be at
          % least 3 output derivatives). also Inputs don't count since their
          % derivatives can be ignored.
          if isempty(objs{k}.numInputDer)
            net.layers(k).numInputDer = max([0, net.layers(k).inputArgPos(inputHasDer)]) ;
          else  % manual override
            net.layers(k).numInputDer = objs{k}.numInputDer ;
          end
        end
      end
      
      % store the sequence of indexes of layers with callable functions
      % (so Inputs and Params, though they hold values/der., are skipped).
      net.order = find(~cellfun('isempty', {net.layers.func})) ;
      
      % store list of non-Param layers, used to clear their derivatives
      net.nonParams = find(cellfun(@(o) ~isa(o, 'Param'), objs)) ;
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
      
      for k = 1:numel(net.layers)
        net.layers(k).value = moveOp(net.layers(k).value) ;
        net.layers(k).der = moveOp(net.layers(k).der) ;
      end
    end
    
    function eval(net, derOutput, accumulateParamDers, lastLayer)
      if nargin < 4
        order = net.order ;
      else  % only evaluate until the specified layer index
        order = net.order(1 : find(net.order == lastLayer, 1)) ;
      end
      
      % use local variable for efficiency
      layers = net.layers ;   %#ok<*PROPLC> % disable MLint's complaints
      net.layers = {} ;  % allow Matlab to release memory in 'layers'
      
      % forward pass
      for k = order
        % populate missing values in the function arguments with the outputs
        % of input layers
        layer = layers(k) ;
        args = layer.args ;
        args(layer.inputArgPos) = {layers(layer.inputLayerIdx).value} ;
        
        % call the layer's function and store the result
        layers(k).value = layer.func(args{:}) ;
      end
      
      % backward pass
      if nargin > 1 && ~isempty(derOutput)
        % clear all derivatives
        if nargin < 3 || ~accumulateParamDers
          [layers.der] = deal(0) ;
        else  % clear all derivatives except for Params'
          [layers(net.nonParams).der] = deal(0) ;
        end
        
        % set root layer's output derivative
        layers(end).der = derOutput ;
        
        for k = order(end:-1:1)
          % populate function arguments with outputs of input layers, and
          % the derivative argument (making the function run in bwd mode)
          layer = layers(k) ;
          args = layer.bwdArgs ;
          inputIdx = layer.inputLayerIdx ;
          
          args(layer.inputArgPos) = {layers(inputIdx).value} ;
          args{layer.derArgPos} = layer.der ;
          
          % call the layer's function in bwd mode
          inputDer = cell(1, layer.numInputDer) ;
          [inputDer{:}] = layer.func(args{:}) ;
          
          % sum derivatives. note some inputDer may be ignored (because
          % they're not input layers, just constant arguments).
          for i = find(layer.inputArgPos <= numel(inputDer))
            layers(inputIdx(i)).der = layers(inputIdx(i)).der + ...
                inputDer{layer.inputArgPos(i)} ;
          end
        end
      end
      
      net.layers = layers ;
    end
    
    
    function value = getValue(net, layer)
      if ~isnumeric(layer)
        assert(isa(layer, 'Layer'), 'LAYER must either be layer indexes or a Layer object.') ;
        layer = layer.idx ;
      end
      if isscalar(layer)
        value = net.layers(layer).value ;
      else
        value = {net.layers(layer).value} ;
      end
    end
    
    
    function der = getDer(net, layer)
      if ~isnumeric(layer)
        assert(isa(layer, 'Layer'), 'LAYER must either be layer indexes or a Layer object.') ;
        layer = layer.idx ;
      end
      if isscalar(layer)
        der = net.layers(layer).der ;
      else
        der = {net.layers(layer).der} ;
      end
    end
    
    function setValue(net, layer, value)
      if ~isnumeric(layer)
        assert(isa(layer, 'Layer'), 'LAYER must either be layer indexes or a Layer object.') ;
        layer = layer.idx ;
      end
      if isscalar(layer)
        net.layers(layer).value = value ;
      else
        [net.layers(layer).value] = deal(value{:}) ;
      end
    end
    
    function setDer(net, layer, der)
      if ~isnumeric(layer)
        assert(isa(layer, 'Layer'), 'LAYER must either be layer indexes or a Layer object.') ;
        layer = layer.idx ;
      end
      if isscalar(layer)
        net.layers(layer).der = der ;
      else
        [net.layers(layer).der] = deal(der{:}) ;
      end
    end
    
    function setInputs(net, varargin)
      for i = 1 : 2 : numel(varargin) - 1
        net.layers(net.inputs.(varargin{i})).value = varargin{i+1} ;
      end
    end
    
    function s = saveobj(net)
      s.conserveMemory = net.conserveMemory ;
      s.layers = net.layers ;
      s.inputs = net.inputs ;
      s.params = net.params ;
      s.nonParams = net.nonParams ;
      s.order = net.order ;
    end
  end
  
  methods (Static)
    function net = loadobj(s)
      net.conserveMemory = s.conserveMemory ;
      net.layers = s.layers ;
      net.inputs = s.inputs ;
      net.params = s.params ;
      net.nonParams = s.nonParams ;
      net.order = s.order ;
    end
  end
end

