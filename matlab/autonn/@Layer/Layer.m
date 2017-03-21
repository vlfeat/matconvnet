classdef Layer < matlab.mixin.Copyable
%Layer
%   The Layer object is the main building block for defining networks in
%   the autonn framework. It specifies a function call in a computational
%   graph.
%
%   Generally there is no need to invoke Layer directly. One can start
%   by defining the network inputs: (Input is a subclass of Layer)
%
%      images = Input() ;
%      labels = Input() ;
%
%   And then composing them using the overloaded functions:
%
%      prediction = vl_nnconv(images, 'size', [1, 1, 4, 3]) ;
%      loss = vl_nnsoftmaxloss(prediction, labels) ;
%
%   Both vl_nnconv and vl_nnsoftmaxloss will not run directly, but are
%   overloaded to return Layer objects that contain the function call
%   information.
%
%   Math operators are also overloaded, making it possible to mix layers
%   with arbitrary mathematical formulas and differentiate through them.
%   See the EXAMPLES/AUTONN/ directory for example usage.
%
%
%   ### Custom Layers ###
%   To create custom layers from a user function FUNC, create a generator:
%
%      customLoss = Layer.fromFunction(@func) ;
%
%   Then compose it normally, like other overloaded MatConvNet functions:
%
%      loss = customLoss(prediction, labels) ;
%
%   FUNC must accept extra output derivative arguments, which will be
%   supplied when in backward mode. In the above example, it will be called
%   as:
%   * Forward mode:   Y = FUNC(X, L)
%   * Backward mode:  DZDX = FUNC(X, L, DZDY)
%   Where DZDY is the output derivative and DZDX is the input derivative.
%
%   Further notes:
%
%   * If your custom function is called with any name-value pairs, the
%     output derivative arguments appear *before* the name-value pairs.
%
%   * The function can return multiple values.
%
%   * If you do not wish to return the derivative for some inputs (e.g.
%     labels), restrict the number of returned derivatives:
%       customLoss = Layer.fromFunction(@func, 'numInputDer', 1) ;
%
%
%   See also NET.

% Copyright (C) 2016 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  properties (Access = public)
    inputs = {}  % list of inputs, either constants or other Layers
  end
  
  properties (SetAccess = public, GetAccess = public)
    func = []  % main function being called
    name = ''  % optional name (for debugging mostly; a layer is a unique handle object that can be passed around)
    numOutputs = []  % to manually specify the number of outputs returned in fwd mode
    numInputDer = []  % to manually specify the number of input derivatives returned in bwd mode
    accumDer = true  % to manually specify that the input derivatives are *not* accumulated. used to implement ReLU short-circuiting.
    meta = []  % optional meta properties
    source = []  % call stack (source files and line numbers) where this Layer was created
    diagnostics = []  % whether to plot the mean, min and max of the Layer's output var. empty for automatic (network outputs only).
    optimize = true  % whether to optimize this Layer, function-dependent (e.g. merge vl_nnwsum)
  end
  
  properties (SetAccess = {?Net}, GetAccess = public)
    outputVar = 0  % index of the output var in a Net, used during its construction
    id = []  % unique ID assigned on creation; does not have to persist, just be different from all other Layers in memory
  end
  
  properties (Access = protected)
    copied = []  % reference of deep copied object, used internally for deepCopy()
    enableCycleChecks = true  % to avoid redundant cycle checks when implicitly calling set.inputs()
  end
  
  methods
    function obj = Layer(func, varargin)  % wrap a function call
      obj.saveStack() ;  % record source file and line number, for debugging
      
      obj.id = Layer.uniqueId() ;  % set unique ID, needed for some operations
      
      if nargin == 0 && (isa(obj, 'Input') || isa(obj, 'Param') || isa(obj, 'Selector'))
        return  % called during Input, Param or Selector construction, nothing to do
      end
      
      % convert from SimpleNN to DagNN
      if isstruct(func) && isfield(func, 'layers')
        func = dagnn.DagNN.fromSimpleNN(func, 'CanonicalNames', true) ;
      end
      
      % convert from DagNN to Layer
      if isa(func, 'dagnn.DagNN')
         obj = dagnn2autonn(func) ;
         if isscalar(obj)
           obj = obj{1} ;
         else  % wrap multiple outputs in a weighted sum
           obj = Layer(@vl_nnwsum, obj{:}, 'weights', ones(1, numel(obj))) ;
         end
         return
      else
        assert(isa(func, 'function_handle'), ...
          'Input must be a function handle, a SimpleNN struct or a DagNN.') ;
      end
      
      assert(isa(func, 'function_handle'), 'Must specify a function handle as the first argument.') ;
      
      obj.enableCycleChecks = false ;
      obj.func = func ;
      obj.inputs = varargin(:)' ;
      
      % call setup function if defined. it can change the inputs list (not
      % allowed for outside functions, to preserve call graph structure).
      obj.inputs = autonn_setup(obj) ;
      obj.enableCycleChecks = true ;
    end
    
    function set.inputs(obj, newInputs)
      if obj.enableCycleChecks
        % must check for cycles, to ensure DAG structure.
        visited = Layer.initializeRecursion() ;
        for i = 1:numel(newInputs)
          if isa(newInputs{i}, 'Layer')
            newInputs{i}.cycleCheckRecursive(obj, visited) ;
          end
        end
      end
      
      obj.inputs = newInputs;
    end
    
    
    % overloaded MatConvNet functions
    function y = vl_nnconv(obj, varargin)
      y = Layer(@vl_nnconv, obj, varargin{:}) ;
    end
    function y = vl_nnconvt(obj, varargin)
      y = Layer(@vl_nnconvt, obj, varargin{:}) ;
    end
    function y = vl_nnpool(obj, varargin)
      y = Layer(@vl_nnpool, obj, varargin{:}) ;
    end
    function y = vl_nnrelu(obj, varargin)
      y = Layer(@vl_nnrelu, obj, varargin{:}) ;
    end
    function y = vl_nnsigmoid(obj, varargin)
      y = Layer(@vl_nnsigmoid, obj, varargin{:}) ;
    end
    function y = vl_nndropout(obj, varargin)
      y = Layer(@vl_nndropout, obj, varargin{:}) ;
    end
    function y = vl_nnbilinearsampler(obj, varargin)
      y = Layer(@vl_nnbilinearsampler, obj, varargin{:}) ;
    end
    function y = vl_nnaffinegrid(obj, varargin)
      y = Layer(@vl_nnaffinegrid, obj, varargin{:}) ;
    end
    function y = vl_nncrop(obj, varargin)
      y = Layer(@vl_nncrop, obj, varargin{:}) ;
    end
    function y = vl_nnnoffset(obj, varargin)
      y = Layer(@vl_nnnoffset, obj, varargin{:}) ;
    end
    function y = vl_nnnormalize(obj, varargin)
      y = Layer(@vl_nnnormalize, obj, varargin{:}) ;
    end
    function y = vl_nnnormalizelp(obj, varargin)
      y = Layer(@vl_nnnormalizelp, obj, varargin{:}) ;
    end
    function y = vl_nnspnorm(obj, varargin)
      y = Layer(@vl_nnspnorm, obj, varargin{:}) ;
    end
    function y = vl_nnbnorm(obj, varargin)
      y = Layer(@vl_nnbnorm, obj, varargin{:}) ;
    end
    function y = vl_nnsoftmax(obj, varargin)
      y = Layer(@vl_nnsoftmax, obj, varargin{:}) ;
    end
    function y = vl_nnpdist(obj, varargin)
      y = Layer(@vl_nnpdist, obj, varargin{:}) ;
    end
    function y = vl_nnsoftmaxloss(obj, varargin)
      y = Layer(@vl_nnsoftmaxloss, obj, varargin{:}) ;
    end
    function y = vl_nnloss(obj, varargin)
      y = Layer(@vl_nnloss, obj, varargin{:}) ;
    end
    function [hn, cn] = vl_nnlstm(obj, varargin)
      [hn, cn] = Layer.createLayer(@vl_nnlstm, [{obj}, varargin]) ;
    end
    
    
    % overloaded native Matlab functions
    function y = reshape(obj, varargin)
      y = Layer(@reshape, obj, varargin{:}) ;
    end
    function y = repmat(obj, varargin)
      y = Layer(@repmat, obj, varargin{:}) ;
    end
    function y = permute(obj, varargin)
      y = Layer(@permute, obj, varargin{:}) ;
    end
    function y = ipermute(obj, varargin)
      y = Layer(@ipermute, obj, varargin{:}) ;
    end
    function y = squeeze(obj, varargin)
      y = Layer(@squeeze, obj, varargin{:}) ;
    end
    function y = size(obj, varargin)
      y = Layer(@size, obj, varargin{:}) ;
    end
    function y = sum(obj, varargin)
      y = Layer(@sum, obj, varargin{:}) ;
    end
    function y = mean(obj, varargin)
      y = Layer(@mean, obj, varargin{:}) ;
    end
    function y = max(obj, varargin)
      y = Layer(@max, obj, varargin{:}) ;
    end
    function y = min(obj, varargin)
      y = Layer(@min, obj, varargin{:}) ;
    end
    function y = abs(obj, varargin)
      y = Layer(@abs, obj, varargin{:}) ;
    end
    function y = sqrt(obj, varargin)
      y = Layer(@sqrt, obj, varargin{:}) ;
    end
    function y = exp(obj, varargin)
      y = Layer(@exp, obj, varargin{:}) ;
    end
    function y = log(obj, varargin)
      y = Layer(@log, obj, varargin{:}) ;
    end
    function y = inv(obj, varargin)
      y = Layer(@inv, obj, varargin{:}) ;
    end
    function y = cat(obj, varargin)
      y = Layer(@cat, obj, varargin{:}) ;
    end
    function y = gpuArray(obj)
      % need to wrap gpuArray so that it is disabled in CPU mode
      y = Layer(@gpuArray_wrapper, obj, Input('gpuMode')) ;
      y.numInputDer = 1 ;
    end
    function y = gather(obj)
      y = Layer(@gather, obj) ;
    end
    
    % overloaded relational and logical operators (no derivative).
    % note: short-circuited scalar operators (&&, ||) cannot be overloaded,
    % use other logical operators instead (&, |).
    
    function y = eq(a, b, same)
      % EQ(A, B), A == B
      % Returns a Layer that tests equality of the outputs of two Layers
      % (one of them may be constant).
      % EQ(A, B, 'sameInstance')
      % Checks if two variables refer to the same Layer instance (i.e.,
      % calls the == operator for handle classes).
      if nargin <= 2
        y = Layer(@eq, a, b) ;
      else
        assert(isequal(same, 'sameInstance'), 'The only accepted extra flag for EQ is ''sameInstance''.') ;
        y = eq@handle(a, b) ;
      end
    end
    function y = ne(a, b)
      y = Layer(@ne, a, b) ;
    end
    function y = lt(a, b)
      y = Layer(@lt, a, b) ;
    end
    function y = gt(a, b)
      y = Layer(@gt, a, b) ;
    end
    function y = le(a, b)
      y = Layer(@le, a, b) ;
    end
    function y = ge(a, b)
      y = Layer(@ge, a, b) ;
    end
    
    function y = and(a, b)
      y = Layer(@and, a, b) ;
    end
    function y = or(a, b)
      y = Layer(@or, a, b) ;
    end
    function y = not(a)
      y = Layer(@not, a) ;
    end
    function y = xor(a, b)
      y = Layer(@xor, a, b) ;
    end
    function y = any(obj, varargin)
      y = Layer(@any, obj, varargin{:}) ;
    end
    function y = all(obj, varargin)
      y = Layer(@all, obj, varargin{:}) ;
    end
    
    % overloaded math operators. any additions, negative signs and scalar
    % factors are merged into a single vl_nnwsum by the Layer constructor.
    % vl_nnbinaryop does singleton expansion, vl_nnmatrixop does not.
    
    function c = plus(a, b)
      c = Layer(@vl_nnwsum, a, b, 'weights', [1, 1]) ;
    end
    function c = minus(a, b)
      c = Layer(@vl_nnwsum, a, b, 'weights', [1, -1]) ;
    end
    function c = uminus(a)
      c = Layer(@vl_nnwsum, a, 'weights', -1) ;
    end
    function c = uplus(a)
      c = a ;
    end
    
    function c = times(a, b)
      % optimization: for simple scalar constants, use a vl_nnwsum layer
      if isnumeric(a) && isscalar(a)
        c = Layer(@vl_nnwsum, b, 'weights', a) ;
      elseif isnumeric(b) && isscalar(b)
        c = Layer(@vl_nnwsum, a, 'weights', b) ;
      else  % general case
        c = Layer(@vl_nnbinaryop, a, b, @times) ;
      end
    end
    function c = rdivide(a, b)
      if isnumeric(b) && isscalar(b)  % optimization for scalar constants
        c = Layer(@vl_nnwsum, a, 'weights', 1 / b) ;
      else
        c = Layer(@vl_nnbinaryop, a, b, @rdivide) ;
      end
    end
    function c = ldivide(a, b)
      if isnumeric(a) && isscalar(a)  % optimization for scalar constants
        c = Layer(@vl_nnwsum, b, 'weights', 1 / a) ;
      else
        c = Layer(@vl_nnbinaryop, a, b, @ldivide) ;
      end
    end
    function c = power(a, b)
      c = Layer(@vl_nnbinaryop, a, b, @power) ;
    end
    
    function y = transpose(a)
      y = Layer(@vl_nnmatrixop, a, [], @transpose) ;
    end
    function y = ctranspose(a)
      y = Layer(@vl_nnmatrixop, a, [], @ctranspose) ;
    end
    
    function c = mtimes(a, b)
      % optimization: for simple scalar constants, use a vl_nnwsum layer
      if isnumeric(a) && isscalar(a)
        c = Layer(@vl_nnwsum, b, 'weights', a) ;
      elseif isnumeric(b) && isscalar(b)
        c = Layer(@vl_nnwsum, a, 'weights', b) ;
      else  % general case
        c = Layer(@vl_nnmatrixop, a, b, @mtimes) ;
      end
    end
    function c = mrdivide(a, b)
      if isnumeric(b) && isscalar(b)  % optimization for scalar constants
        c = Layer(@vl_nnwsum, a, 'weights', 1 / b) ;
      else
        c = Layer(@vl_nnmatrixop, a, b, @mrdivide) ;
      end
    end
    function c = mldivide(a, b)
      if isnumeric(a) && isscalar(a)  % optimization for scalar constants
        c = Layer(@vl_nnwsum, b, 'weights', 1 / a) ;
      else
        c = Layer(@vl_nnmatrixop, a, b, @mldivide) ;
      end
    end
    function c = mpower(a, b)
      c = Layer(@vl_nnmatrixop, a, b, @mpower) ;
    end
    
    function y = vertcat(obj, varargin)
      y = Layer(@cat, 1, obj, varargin{:}) ;
    end
    function y = horzcat(obj, varargin)
      y = Layer(@cat, 2, obj, varargin{:}) ;
    end
    
    function y = colon(obj, varargin)
      y = Layer(@colon, obj, varargin{:}) ;
    end
    
    % overloaded indexing
    function varargout = subsref(a, s)
      if strcmp(s(1).type, '()')
        varargout{1} = Layer(@autonn_slice, a, s.subs{:}) ;
      else
        [varargout{1:nargout}] = builtin('subsref', a, s) ;
      end
    end
    
    % overload END keyword, e.g. X(1:end-1). see DOC OBJECT-END-INDEXING.
    % a difficult choice: returning a constant size requires knowing all
    % input sizes in advance (i.e. to call evalOutputSize). returning a
    % Layer (on-the-fly size calculation) has overhead and also requires
    % overloading the colon (:) operator.
    function idx = end(obj, dim, ndim)
      error('Not supported, use SIZE(X,DIM) or a constant size instead.') ;
    end
  end
  
  methods (Access = {?Net, ?Layer})
    function cycleCheckRecursive(obj, root, visited)
      if eq(obj, root, 'sameInstance')
        error('MatConvNet:CycleCheckFailed', 'Input assignment creates a cycle in the network.') ;
      end
      
      % recurse on inputs
      idx = obj.getNextRecursion(visited) ;
      for i = idx
        obj.inputs{i}.cycleCheckRecursive(root, visited) ;
      end
      visited(obj.id) = true ;
    end
    
    function idx = getNextRecursion(obj, visited)
      % Used by findRecursive, cycleCheckRecursive, deepCopyRecursive, etc,
      % to avoid redundant recursions in very large networks.
      % Returns indexes of inputs to recurse on, that have not been visited
      % yet during this operation. The list of layers seen so far is
      % managed efficiently with the dictionary VISITED.
      
      valid = false(1, numel(obj.inputs)) ;
      for i = 1:numel(obj.inputs)
        if isa(obj.inputs{i}, 'Layer')
          valid(i) = ~visited.isKey(obj.inputs{i}.id) ;
        end
      end
      idx = find(valid) ;
    end
    
    function saveStack(obj)
      % record call stack (source files and line numbers), starting with
      % the first function in user-land (not part of autonn).
      stack = dbstack('-completenames') ;
      
      % current file's directory (e.g. <MATCONVNET>/matlab/autonn)
      p = [fileparts(stack(1).file), filesep] ;
      
      % find a non-matching directory (i.e., not part of autonn directly)
      for i = 2:numel(stack)
        if ~strncmp(p, stack(i).file, numel(p))
          obj.source = stack(i:end) ;
          return
        end
      end
      obj.source = struct('file',{}, 'name',{}, 'line',{}) ;
    end
  end
  
  methods (Static)
    function generator = fromFunction(func, varargin)
      % Returns a layer generator, based on a custom function FUNC.
      % May set additional properties as name-value pairs (numInputDer).
      assert(isa(func, 'function_handle'), 'Argument must be a valid function handle.') ;
      
      opts = varargin ;
      generator = @(varargin) Layer.createLayer(func, varargin, opts{:}) ;
    end
    
    function varargout = createLayer(func, args, varargin)
      % Create a layer with given function handle FUNC and arguments
      % cell array ARGS, optionally setting additional properties as
      % name-value pairs (numInputDer). numOutputs is inferred.
      % Supports multiple outputs.
      assert(isa(func, 'function_handle'), 'Argument must be a valid function handle.') ;
      
      opts.numInputDer = [] ;
      opts.numOutputs = [] ;
      opts = vl_argparse(opts, varargin) ;
      
      % main output
      varargout = cell(1, nargout) ;
      varargout{1} = Layer(func, args{:}) ;
      varargout{1}.numOutputs = nargout ;  % infer number of layer outputs from this function call
      varargout{1}.numInputDer = opts.numInputDer ;
      
      % selectors for any additional outputs
      for i = 2:nargout
        varargout{i} = Selector(varargout{1}, i) ;
      end
    end
    
    function workspaceNames(modifier)
      % LAYER.WORKSPACENAMES()
      % Sets layer names based on the name of the corresponding variables
      % in the caller's workspace. Only empty names are set.
      %
      % LAYER.WORKSPACENAMES(MODIFIER)
      % Specifies a function handle to be evaluated on each name, possibly
      % modifying it (e.g. append a prefix or suffix).
      %
      % See also SEQUENTIALNAMES.
      %
      % Example:
      %    images = Input() ;
      %    Layer.workspaceNames() ;
      %    >> images.name
      %    ans =
      %       'images'
      
      if nargin < 1, modifier = @deal ; end
      
      varNames = evalin('caller','who') ;
      for i = 1:numel(varNames)
        layer = evalin('caller', varNames{i}) ;
        if isa(layer, 'Layer') && isempty(layer.name)
          layer.name = modifier(varNames{i}) ;
        end
      end
    end
    
    function setDiagnostics(obj, value)
      if iscell(obj)  % applies recursively to nested cell arrays
        for i = 1:numel(obj)
          Layer.setDiagnostics(obj{i}, value) ;
        end
      else
        obj.diagnostics = value ;
      end
    end
    
    % overloaded native Matlab functions, static (first argument is not a
    % Layer object, call with Layer.rand(...)).
    function y = rand(obj, varargin)
      y = Layer(@rand, obj, varargin{:}) ;
    end
    function y = randi(obj, varargin)
      y = Layer(@randi, obj, varargin{:}) ;
    end
    function y = randn(obj, varargin)
      y = Layer(@randn, obj, varargin{:}) ;
    end
    function y = randperm(obj, varargin)
      y = Layer(@randperm, obj, varargin{:}) ;
    end
    function y = zeros(obj, varargin)
      y = Layer(@zeros, obj, varargin{:}) ;
    end
    function y = ones(obj, varargin)
      y = Layer(@ones, obj, varargin{:}) ;
    end
    function y = inf(obj, varargin)
      y = Layer(@inf, obj, varargin{:}) ;
    end
    function y = nan(obj, varargin)
      y = Layer(@nan, obj, varargin{:}) ;
    end
    function y = eye(obj, varargin)
      y = Layer(@eye, obj, varargin{:}) ;
    end
  end
  
  methods (Static, Access = 'private')
    function id = uniqueId()
      persistent nextId
      if isempty(nextId)
        nextId = uint32(1) ;
      end
      id = nextId ;
      nextId = nextId + 1 ;
    end
    
    function visited = initializeRecursion()
      % See getNextRecursion
      visited = containers.Map('KeyType','uint32', 'ValueType','any') ;
    end
  end
end

