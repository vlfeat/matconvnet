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
    test = []  % test mode function calls
    vars = {}  % cell array of variables and their derivatives
    inputs = []  % struct of network's Inputs, indexed by name
    params = []  % list of Params
  end
  properties (SetAccess = public, GetAccess = public)
    meta = []  % optional meta properties
    diagnostics = []  % list of diagnosed vars (see Net.plotDiagnostics)
  end

  methods  % methods defined in their own files
    eval(net, mode, derOutput, accumulateParamDers)
    plotDiagnostics(net, numPoints)
    displayVars(net, vars)
  end
  methods (Access = private)
    build(net, varargin)
  end
  
  methods
    function net = Net(varargin)
      % constructors can't be defined in external files, so use a method
      net.build(varargin{:}) ;
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
    
    function value = getValue(net, var)
      if ~isnumeric(var)
        assert(isa(var, 'Layer'), 'VAR must either be var indexes or a Layer object.') ;
        var = var.outputVar ;
      end
      if isscalar(var)
        value = net.vars{var} ;
      else
        value = net.vars(var) ;
      end
    end
    
    
    function der = getDer(net, var)
      if ~isnumeric(var)
        assert(isa(var, 'Layer'), 'VAR must either be var indexes or a Layer object.') ;
        var = var.outputVar ;
      end
      if isscalar(var)
        der = net.vars{var + 1} ;
      else
        der = net.vars(var + 1) ;
      end
    end
    
    function setValue(net, var, value)
      if ~isnumeric(var)
        assert(isa(var, 'Layer'), 'VAR must either be var indexes or a Layer object.') ;
        var = var.outputVar ;
      end
      if isscalar(var)
        net.vars{var} = value ;
      else
        net.vars(var) = value ;
      end
    end
    
    function setDer(net, var, der)
      if ~isnumeric(var)
        assert(isa(var, 'Layer'), 'VAR must either be var indexes or a Layer object.') ;
        var = var.outputVar ;
      end
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
        net.vars{net.inputs.(varargin{i})} = varargin{i+1} ;
      end
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
      s.inputs = net.inputs ;
      s.params = net.params ;
      s.meta = net.meta ;
      s.diagnostics = net.diagnostics ;
      
      % only save var contents corresponding to parameters, all other vars
      % are transient
      s.vars = cell(size(net.vars)) ;
      s.vars([net.params.var]) = net.vars([net.params.var]) ;
    end
    
    function loadobj(net, s)
      net.forward = s.forward ;
      net.backward = s.backward ;
      net.test = s.test ;
      net.vars = s.vars ;
      net.inputs = s.inputs ;
      net.params = s.params ;
      net.meta = s.meta ;
      net.diagnostics = s.diagnostics ;
    end
  end
  
  methods (Static, Access = private)
    function layer = parseArgs(layer, args)
      % helper function to parse a layer's arguments, storing the constant
      % arguments (args), non-constant var indexes (inputVars), and their
      % positions in the arguments list (inputArgPos).
      inputVars = [] ;
      inputArgPos = [] ;
      for a = 1:numel(args)
        if isa(args{a}, 'Layer')
          inputVars(end+1) = args{a}.outputVar ;  %#ok<*AGROW>
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

