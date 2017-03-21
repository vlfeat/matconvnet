classdef Net < handle
%Net
%   A compiled network, ready to be evaluated on some data.
%
%   While Layer objects are used to easily define a network topology
%   (build-time), a Net object compiles them to a format that can be
%   executed quickly (run-time).
%
%   Example:
%      % define topology
%      images = Input() ;
%      labels = Input() ;
%      prediction = vl_nnconv(images, 'size', [5, 5, 1, 3]) ;
%      loss = vl_nnsoftmaxloss(prediction, labels) ;
%
%      % assign names automatically
%      Layer.workspaceNames() ;
%
%      % compile network
%      net = Net(loss) ;
%
%      % set input data and evaluate network
%      net.setInputs('images', randn(5, 5, 1, 3, 'single'), ...
%                    'labels', single(1:3)) ;
%      net.eval() ;
%
%      disp(net.getValue(loss)) ;
%      disp(net.getDer(images)) ;
%
%   Note: The examples cannot be ran in the <MATCONVNET>/matlab folder
%   due to function shadowing.
%
%   See also LAYER.

% Copyright (C) 2016 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  properties (SetAccess = protected, GetAccess = public)
    forward = []  % forward pass function calls
    backward = []  % backward pass function calls
    vars = {}  % cell array of variables and their derivatives
    inputs = []  % struct of network's Inputs, indexed by name
    params = []  % list of Params
    gpu = false  % whether the network is in GPU or CPU mode
    isGpuVar = []  % whether each variable or derivative can be on the GPU
  end
  properties (SetAccess = public, GetAccess = public)
    meta = []  % optional meta properties
    diagnostics = []  % list of diagnosed vars (see Net.plotDiagnostics)
  end

  methods  % methods defined in their own files
    eval(net, mode, derOutput, accumulateParamDers)
    plotDiagnostics(net, numPoints)
    displayVars(net, vars, varargin)
  end
  methods (Access = private)
    build(net, varargin)
    optimizeVars(net, opts, objs)
  end
  
  methods
    function net = Net(varargin)
      % load from struct, distinguishing from SimpleNN
      if isscalar(varargin) && isstruct(varargin{1}) && ~isfield(varargin{1}, 'layers')
        net = Net.loadobj(varargin{1}) ;
        return
      end

      if isscalar(varargin) && ~isa(varargin{1}, 'Layer')
        % convert SimpleNN or DagNN to Layer
        s = varargin{1} ;
        if isstruct(s) && isfield(s, 'layers')
          s = dagnn.DagNN.fromSimpleNN(s, 'CanonicalNames', true) ;
        end
        if isa(s, 'dagnn.DagNN')
          s = dagnn2autonn(s) ;
        end
        if iscell(s)
          varargin = s ;  % varargin should contain a list of Layer objects
        else
          varargin = {s} ;
        end
      end

      % build Net from a list of Layers
      net.build(varargin{:}) ;
    end
    
    function move(net, device)
    %MOVE Move data to CPU or GPU
    %  MOVE(DESTINATION) moves the data associated to the net object OBJ
    %  to either the 'gpu' or the 'cpu'.
      switch device
        case 'gpu'
          moveOp = @gpuArray ;
        case 'cpu'
          moveOp = @gather ;
          
           % by moving to the CPU we lose the knowledge of which vars are
           % supposed to be on the GPU, so store that. once on the GPU,
           % always on the GPU.
          net.isGpuVar = net.isGpuVar | cellfun('isclass', net.vars, 'gpuArray') ;
        otherwise
          error('Unknown device ''%s''.', device) ;
      end
      
      net.vars(net.isGpuVar) = cellfun(moveOp, net.vars(net.isGpuVar), 'UniformOutput',false) ;
      
      net.gpu = strcmp(device, 'gpu') ;
      if isfield(net.inputs, 'gpuMode')
        net.setInputs('gpuMode', net.gpu) ;
      end
    end
    
    function value = getValue(net, var)
      %NET.GETVALUE(VAR)
      % Returns the value of a given variable. VAR may either be a layer
      % name or Layer object (referring to its output), or a var index.
      var = net.getVarIndex(var) ;
      if isscalar(var)
        value = net.vars{var} ;
      else
        value = net.vars(var) ;
      end
    end
    
    
    function der = getDer(net, var)
      %NET.GETDER(VAR)
      % Returns the derivative of a variable. VAR may either be a layer
      % name or Layer object (referring to its output), or a var index.
      var = net.getVarIndex(var) ;
      if isscalar(var)
        der = net.vars{var + 1} ;
      else
        der = net.vars(var + 1) ;
      end
    end
    
    function setValue(net, var, value)
      %NET.SETVALUE(VAR, VALUE)
      % Sets the value of a given variable. VAR may either be a layer name
      % or Layer object (referring to its output), or a var index.
      var = net.getVarIndex(var) ;
      if isscalar(var)
        net.vars{var} = value ;
      else
        net.vars(var) = value ;
      end
    end
    
    function setDer(net, var, der)
      %NET.SETDER(VAR)
      % Sets the derivative of a variable. VAR may either be a layer name
      % or Layer object (referring to its output), or a var index.
      var = net.getVarIndex(var) ;
      if isscalar(var)
        net.vars{var + 1} = der ;
      else
        net.vars(var + 1) = der ;
      end
    end
    
    function setInputs(net, varargin)
      assert(mod(numel(varargin), 2) == 0, ...
        'Arguments must be in the form INPUT1, VALUE1, INPUT2, VALUE2,...'),
      
      for i = 1 : 2 : numel(varargin) - 1
        var = net.inputs.(varargin{i}) ;
        value = varargin{i+1} ;
        if net.gpu && net.isGpuVar(var)  % move to GPU if needed
          value = gpuArray(value) ;
        end
        net.vars{var} = value ;
      end
    end
    
    function idx = getVarIndex(net, var, errorIfNotFound)
      if nargin < 3
        errorIfNotFound = true ;
      end
      if ischar(var)
        % search for var/layer by name
        if isfield(net.inputs, var)  % search inputs
          idx = net.inputs.(var) ;
        else  % search params
          param = strcmp({net.params.name}, var) ;
          if any(param)
            idx = net.params(param).var ;
          else  % search layers
            layer = strcmp({net.forward.name}, var) ;
            if any(layer)
              idx = net.forward(layer).outputVar ;
            else
              if errorIfNotFound,
                error(['No var with specified name ''' var '''.']) ;
              end
              idx = 0 ;
            end
          end
        end
      elseif isa(var, 'Layer')
        idx = var.outputVar(1) ;
      else
        assert(isnumeric(var), 'VAR must either be a layer name, a Layer object, or var indexes.') ;
        idx = var ;
      end
    end
    
    function display(net, name)
      if nargin < 2
        name = inputname(1) ;
      end
      fprintf('\n%s = \n', name) ;
      disp(net) ;
      
      if ~isempty(name)
        if ~isempty(net.vars)
          fprintf('  <a href="matlab:%s.displayVars()">Display variables</a>\n\n', name) ;
        else
          fprintf(['  <a href="matlab:%s.displayVars(vars)">Display variables ' ...
            '(NOTE: Must be in the scope of NET.EVAL; use DBUP/DBDOWN)</a>\n\n'], name) ;
        end
      end
    end
    
    function s = saveobj(net)
      s.forward = net.forward ;
      s.backward = net.backward ;
      s.inputs = net.inputs ;
      s.params = net.params ;
      s.gpu = net.gpu ;
      s.isGpuVar = net.isGpuVar ;
      s.meta = net.meta ;
      s.diagnostics = net.diagnostics ;
      
      % only save var contents corresponding to parameters, all other vars
      % are transient
      s.vars = cell(size(net.vars)) ;
      s.vars([net.params.var]) = net.vars([net.params.var]) ;
    end
  end
  
  methods (Static, Access = private)
    function net = loadobj(s)
      net = Net() ;
      net.forward = s.forward ;
      net.backward = s.backward ;
      net.vars = s.vars ;
      net.inputs = s.inputs ;
      net.params = s.params ;
      net.gpu = s.gpu ;
      net.isGpuVar = s.isGpuVar ;
      net.meta = s.meta ;
      net.diagnostics = s.diagnostics ;
    end
    
    function layer = parseArgs(layer, args)
      % helper function to parse a layer's arguments, storing the constant
      % arguments (args), non-constant var indexes (inputVars), and their
      % positions in the arguments list (inputArgPos).
      inputVars = [] ;
      inputArgPos = [] ;
      for a = 1:numel(args)
        if isa(args{a}, 'Layer')
          % note only the first output is taken if there's more than one;
          % other outputs are reached using Selectors
          inputVars(end+1) = args{a}.outputVar(1) ;  %#ok<*AGROW>
          inputArgPos(end+1) = a ;
          args{a} = [] ;
        end
      end
      layer.args = args ;
      layer.inputVars = inputVars ;
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

