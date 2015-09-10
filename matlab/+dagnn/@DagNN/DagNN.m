classdef DagNN < handle
%DagNN Directed acyclic graph neural network
%   DagNN is a CNN wrapper alternative to SimpleNN. It is object
%   oriented and allows constructing networks with a directed acyclic
%   graph (DAG) topology. It is therefore far more flexible, although
%   a little more complex and slightly slower for small CNNs.
%
%   A DAG object contains the following data members:
%
%   - `layers`: The network layers.
%   - `vars`: The network variables.
%   - `params`: The network parameters.
%   - `meta`: Additional information relative to the CNN (e.g. input
%      image format specification).
%
%   There are additional transient data members:
%
%   `mode`:: [`normal`]
%      This flag can either be `normal` or `test`. In the latter case,
%      certain blocks switch to a test mode suitable for validation or
%      evaluation as opposed to training. For instance, dropout
%      becomes a pass-through block in `test` mode.
%
%   `paramDersAccumulate`:: [`false`]
%      If this flag is set to `true`, then the derivatives of the
%      network parameters are accumulated rather than rewritten the
%      next time the derivatives are computed.
%
%   `conserveMemory`:: [`true`]
%      If this flag is set to `true`, the DagNN will discard
%      intermediate variable values as soon as they are not needed
%      anymore in the calculations. This is particularly important to
%      save memory on GPUs.
%
%   `device`:: [`cpu`]
%      This flag tells whether the DagNN resides in CPU or GPU
%      memory. Use the `DagNN.move()` function to move the DagNN
%      between devices.

% Copyright (C) 2015 Karel Lenc and Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  properties
    layers
    vars
    params
    meta
  end

  properties (Transient)
    mode = 'normal'
    paramDersAccumulate = false
    conserveMemory = true
  end

  properties (Transient, SetAccess = private, GetAccess = public)
    device = 'cpu' ;
  end

  properties (Transient, Access = {?dagnn.DagNN, ?dagnn.Layer}, Hidden = true)
    numPendingVarRefs
    computingDerivative = false
  end

  properties (Transient, Access = private, Hidden = true)
    modifed = false
    varNames = struct()
    paramNames = struct()
    layerNames = struct()
    layerIndexes = {}
  end

  methods
    function obj = DagNN()
    %DAGNN  Initialize an empty DaG
    %   OBJ = DAGNN() initializes an empty DaG.
    %
    %   See Also addLayer
      obj.vars = struct(...
        'name', {}, ...
        'value', {}, ...
        'der', {}, ...
        'fanin', {}, ...
        'fanout', {}, ...
        'precious', {}) ;
      obj.params = struct(...
        'name', {}, ...
        'value', {}, ...
        'der', {}, ...
        'fanout', {}, ...
        'learningRate', {}, ...
        'weightDecay', {}) ;
      obj.layers = struct(...
        'name', {}, ...
        'inputs', {}, ...
        'outputs', {}, ...
        'params', {}, ...
        'inputIndexes', {}, ...
        'outputIndexes', {}, ...
        'paramIndexes', {}, ...
        'block', {}) ;
    end

    function set.mode(obj, mode)
      switch lower(mode)
        case {'normal', 'train'}
          obj.mode = 'normal' ;
        case {'test'}
          obj.mode = 'test' ;
      end
    end

    % Manage the DagNN
    reset(obj)
    move(obj, direction)
    s = saveobj(obj)

    % Manipualte the DagNN
    addLayer(obj, name, block, inputs, outputs, params)
    removeLayer(obj, name)
    rebuild(obj)

    % Process data with the DagNN
    initParams(obj)
    eval(obj, inputs, derOutputs)

    % Get information about the DagNN
    varSizes = getVarSizes(obj, inputSizes)

    % ---------------------------------------------------------------------
    %                                                           Access data
    % ---------------------------------------------------------------------

    function inputs = getInputs(obj)
    %GETINPUTS Get the names of the input variables
    %   INPUTS = GETINPUTS(obj) returns a cell array containing the name
    %   of the input variables of the DaG obj, i.e. the sources of the
    %   DaG (excluding the network parameters, which can also be
    %   considered sources).
      v = find([obj.vars.fanin]==0) ;
      inputs = {obj.vars(v).name} ;
    end

    function outputs = getOutputs(obj)
    %GETOUTPUTS Get the names of the output variables
    %    OUTPUT = GETOUTPUTS(obj) returns a cell array containing the name
    %    of the output variables of the DaG obj, i.e. the sinks of the
    %    DaG.
      v = find([obj.vars.fanout]==0) ;
      outputs = {obj.vars(v).name} ;
    end

    function v = getVarIndex(obj, name)
    %GETVARINDEX Get the index of a variable
    %   INDEX = GETVARINDEX(obj, NAME) obtains the index of the variable
    %   with the specified NAME. NAME can also be a cell array of
    %   strings. If no variable with such a name is found, the value
    %   NaN is returned for the index.
    %
    %   Variables can then be accessed as the `obj.vars(INDEX)`
    %   property of the DaG.
    %
    %   Indexes are stable unless the DaG is modified (e.g. by adding
    %   or removing layers); hence they can be cached for faster
    %   variable access.
    %
    %    See Also getParamIndex
      if iscell(name)
        v = zeros(1, numel(name)) ;
        for k = 1:numel(name)
          v(k) = obj.getVarIndex(name{k}) ;
        end
      else
        if isfield(obj.varNames, name)
          v = obj.varNames.(name) ;
        else
          v = NaN ;
        end
      end
    end

    function p = getParamIndex(obj, name)
    %GETPARAMINDEX Get the index of a parameter
    %   INDEX = GETPARAMINDEX(obj, NAME) obtains the index of the
    %   parameter with the specified NAME. NAME can also be a cell
    %   array of strings. If no parameter with such a name is found,
    %   the value NaN is returned for the index.
    %
    %   Parameters can then be accessed as the `obj.params(INDEX)`
    %   property of the DaG.
    %
    %   Indexes are stable unless the DaG is modified (e.g. by adding
    %   or removing layers); hence they can be cached for faster
    %   parameter access.
    %
    %   See Also getVarIndex
      if iscell(name)
        p = zeros(1, numel(name)) ;
        for k = 1:numel(name)
          p(k) = obj.getParamIndex(name{k}) ;
        end
      else
        if isfield(obj.paramNames, name)
          p = obj.paramNames.(name) ;
        else
          p = NaN ;
        end
      end
    end
  end

  methods (Static)
    obj = fromSimpleNN(net)
    obj = loadobj(s)
  end

  methods (Access = private)
    function v = addVar(obj, name)
    %ADDVAR  Add a variable to the DaG
    %   V = ADDVAR(obj, NAME) adds a varialbe with the specified
    %   NAME to the DaG. This is an internal function; variables
    %   are automatically added when adding layers to the network.
      v = obj.getVarIndex(name) ;
      if ~isnan(v), return ; end
      v = numel(obj.vars) + 1 ;
      obj.vars(v) = struct(...
        'name', {name}, ...
        'value', {[]}, ...
        'der', {[]}, ...
        'fanin', {0}, ...
        'fanout', {0}, ...
        'precious', {false}) ;
      obj.varNames.(name) = v ;
    end

    function p = addParam(obj, name)
    %ADDPARAM  Add a parameter to the DaG
    %   V = ADDPARAM(obj, NAME) adds a parameter with the specified NAME
    %   to the DaG. This is an internal function; parameters are
    %   automatically added when adding layers to the network.
      p = obj.getParamIndex(name) ;
      if ~isnan(p), return ; end
      p = numel(obj.params) + 1 ;
      obj.params(p) = struct(...
        'name', {name}, ...
        'value', {[]}, ...
        'der', {[]}, ...
        'fanout', {0}, ...
        'learningRate', {1}, ...
        'weightDecay', {1}) ;
      obj.paramNames.(name) = p ;
    end
  end
end
