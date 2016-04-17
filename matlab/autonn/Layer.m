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
%   See also NET.

% Copyright (C) 2016 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  properties (SetAccess = private, GetAccess = public)  % not setable after construction, to ensure a proper call graph (no cycles)
    inputs = {}  % list of inputs, either constants or other Layers
    testInputs = 'same'  % list of inputs used in test mode, may be different
  end
  
  properties (SetAccess = public, GetAccess = public)
    func = []  % main function being called
    testFunc = []  % function called in test mode (empty to use the same as in normal mode; 'none' to disable, e.g. dropout)
    name = []  % optional name (for debugging mostly; a layer is a unique handle object that can be passed around)
    numInputDer = []  % to manually specify the number of input derivatives returned in bwd mode
    meta = []  % optional meta properties
  end
  
  properties (SetAccess = {?Net}, GetAccess = public)
    outputVar = 0  % index of the output var in a Net, used during its construction
  end
  
  properties (Access = protected)
    copied = []  % reference of deep copied object, used internally for deepCopy()
  end
  
  methods
    function obj = Layer(func, varargin)  % wrap a function call
      if nargin == 0 && (isa(obj, 'Input') || isa(obj, 'Param'))
        return  % called during Input or Param construction, nothing to do
      end
      
      % convert from SimpleNN to DagNN
      if isstruct(func) && isfield(func, 'layers')
        func = dagnn.DagNN.fromSimpleNN(func, 'CanonicalNames', true) ;
      end
      
      % convert from DagNN to Layer
      if isa(func, 'dagnn.DagNN')
         obj = dagnn2layer(func) ;
         return
      else
        assert(isa(func, 'function_handle'), ...
          'Input must be a function handle, a SimpleNN struct or a DagNN.') ;
      end
      
      assert(isa(func, 'function_handle'), 'Must specify a function handle as the first argument.') ;
      
      obj.func = func ;
      obj.inputs = varargin(:)' ;
      
      % call setup function if defined. it can change the inputs list (not
      % allowed for outside functions, to preserve call graph structure).
      [obj.inputs, obj.testInputs] = autonn_setup(obj) ;
      
    end
    
    function objs = find(obj, what, n, objs)
      % OBJS = OBJ.FIND()
      % OBJS = OBJ.FIND(NAME/FUNC/CLASS)
      % Finds layers, starting at the given output layer. The search
      % criteria can be a layer name, a function handle, or a class name
      % (such as 'Input' or 'Param').
      % By default a cell array is returned, which may be empty.
      %
      % OBJS = OBJ.FIND(..., N)
      % Returns only the Nth object that fits the criteria, in the order of
      % a forward pass (e.g. from the first layer). If N is negative, it is
      % found in the order of a backward pass (e.g. from the last layer,
      % which corresponds to N = -1).
      % Raises an error if no object is found.
      
      if nargin < 2, what = [] ;end
      if nargin < 3, n = 0 ; end
      if nargin < 4, objs = {} ; end
      
      % 'what' is defined, but it may be just N
      if nargin == 2 && isnumeric(what)
        n = what ;
        what = [] ;
      end
      
      if n == 0 || numel(objs) < abs(n)
        % recurse on inputs not on the list yet (when in forward order)
        if n >= 0
          for i = 1:numel(obj.inputs)
            if isa(obj.inputs{i}, 'Layer') && ~any(cellfun(@(o) isequal(obj.inputs{i}, o), objs))
              objs = obj.inputs{i}.find(what, n, objs) ;
            end
          end
        end
        
        % add self to list if it matches the pattern
        if ischar(what)
          if isequal(obj.name, what) || isa(obj, what)
            objs{end+1} = obj ;
          end
        elseif isequal(obj.func, what)
          objs{end+1} = obj ;
        end
        
        % recurse on inputs not on the list yet (when in backward order)
        if n < 0
          for i = 1:numel(obj.inputs)
            if isa(obj.inputs{i}, 'Layer') && ~any(cellfun(@(o) isequal(obj.inputs{i}, o), objs))
              objs = obj.inputs{i}.find(what, n, objs) ;
            end
          end
        end
      end
      
      if nargin < 4 && n ~= 0
        % at the end of the original call, choose the Nth object
        assert(numel(objs) >= abs(n), 'Cannot find a layer fitting the specified criteria.')
        objs = objs{abs(n)} ;
      end
    end
    
    function other = deepCopy(obj, varargin)
      % OTHER = OBJ.DEEPCOPY(SHAREDLAYER1, SHAREDLAYER2, ...)
      % OTHER = OBJ.DEEPCOPY({SHAREDLAYER1, SHAREDLAYER2, ...})
      % Returns a deep copy of a layer, excluding SHAREDLAYER1,
      % SHAREDLAYER2, etc, which are optional. This can be used to
      % implement shared Params, or define the boundaries of the deep copy.
      %
      % To create a shallow copy, use OTHER = OBJ.COPY().
      
      if isscalar(varargin) && iscell(varargin{1})
        varargin = varargin{1} ;
      end
      obj.deepCopyReset() ;
      other = obj.deepCopyRecursive(varargin) ;
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
    
    
    % overloaded native Matlab functions
    function y = reshape(obj, varargin)
      y = Layer(@reshape, obj, varargin{:}) ;
    end
    function y = permute(obj, varargin)
      y = Layer(@permute, obj, varargin{:}) ;
    end
    function y = ipermute(obj, varargin)
      y = Layer(@ipermute, obj, varargin{:}) ;
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
    function y = sqrt(obj, varargin)
      y = Layer(@sqrt, obj, varargin{:}) ;
    end
    function y = cat(obj, varargin)
      y = Layer(@cat, obj, varargin{:}) ;
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
      % optimization: for simple scalar factors, use a vl_nnwsum layer
      if isnumeric(a) && isscalar(a)
        c = Layer(@vl_nnwsum, b, 'weights', a) ;
      elseif isnumeric(b) && isscalar(b)
        c = Layer(@vl_nnwsum, a, 'weights', b) ;
      else  % general case
        c = Layer(@vl_nnbinaryop, a, b, @times) ;
      end
    end
    function c = rdivide(a, b)
      c = Layer(@vl_nnbinaryop, a, b, @rdivide) ;
    end
    function c = ldivide(a, b)
      c = Layer(@vl_nnbinaryop, a, b, @ldivide) ;
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
      % optimization: for simple scalar factors, use a vl_nnwsum layer
      if isnumeric(a) && isscalar(a)
        c = Layer(@vl_nnwsum, b, 'weights', a) ;
      elseif isnumeric(b) && isscalar(b)
        c = Layer(@vl_nnwsum, a, 'weights', b) ;
      else  % general case
        c = Layer(@vl_nnmatrixop, a, b, @mtimes) ;
      end
    end
    function c = mrdivide(a, b)
      c = Layer(@vl_nnmatrixop, a, b, @mrdivide) ;
    end
    function c = mldivide(a, b)
      c = Layer(@vl_nnmatrixop, a, b, @mldivide) ;
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
    
    % overloaded indexing
    function varargout = subsref(a, s)
      if strcmp(s(1).type, '()')
        varargout{1} = Layer(@slice, a, s.subs) ;
      else
        [varargout{1:nargout}] = builtin('subsref', a, s) ;
      end
    end
    
    function display(obj, name)
      % show hyperlinks in command window, allowing one to interactively
      % traverse the network. note that the builtin disp is unchanged.
      
      if nargin < 2
        name = inputname(1) ;
      end
      fprintf('\n%s', name) ;
      
      if ~isempty(name) && ~isempty(obj.inputs)
        if numel(name) > 30  % line break for long names
          fprintf('\n')
        end
        fprintf(' = %s(', char(obj.func)) ;
        
        for i = 1:numel(obj.inputs)
          input = obj.inputs{i} ;
          
          if ~isa(input, 'Layer')
            % use Matlab's native display of single cells, which provides a
            % nice short representation of any object (e.g. '[3x3 double]')
            fprintf(strtrim(evalc('disp({input})'))) ;
          else
            % another layer, display it along with a navigation hyperlink
            if ~isempty(input.name)
              label = input.name ;
            elseif isa(input, 'Input')
              label = 'Input' ;
            elseif isa(input, 'Param')
              label = sprintf('Param(%s)', strtrim(evalc('disp({input.value})'))) ;
            else
              label = sprintf('inputs{%i}', i) ;
            end
            cmd = sprintf('%s.inputs{%i}', name, i) ;

            fprintf('<a href="matlab:display(%s,''%s'')">%s</a>', cmd, cmd, label) ;
          end
          if i < numel(obj.inputs)
            fprintf(', ') ;
          end
        end
        fprintf(')') ;
      else
        fprintf(' = ') ;
      end
      fprintf('\n\n') ;
      
      disp(obj) ;
    end
  end
  
  methods (Access = {?Net, ?Layer})
    function resetOrder(obj)
      obj.outputVar = 0 ;
      for i = 1:numel(obj.inputs)  % recurse on inputs
        if isa(obj.inputs{i}, 'Layer')
          obj.inputs{i}.resetOrder() ;
        end
      end
    end
    
    function layers = buildOrder(obj, layers)
      % recurse on inputs with unassigned indexes (not on the list yet)
      for i = 1:numel(obj.inputs)
        if isa(obj.inputs{i}, 'Layer') && obj.inputs{i}.outputVar == 0
          layers = obj.inputs{i}.buildOrder(layers) ;
        end
      end
      
      % add self to the execution order, after all inputs are there
      layers{end+1} = obj ;
      
      % assign an output var sequentially, leaving slots for derivatives
      obj.outputVar = numel(layers) * 2 - 1 ;
    end
    
    function deepCopyReset(obj)
      obj.copied = [] ;
      for i = 1:numel(obj.inputs)  % recurse on inputs
        if isa(obj.inputs{i}, 'Layer')
          obj.inputs{i}.deepCopyReset() ;
        end
      end
    end
    
    function other = deepCopyRecursive(obj, shared)
      % create a shallow copy first
      other = obj.copy() ;
      
      % pointer to the copied object, to be reused by any subsequent deep
      % copied layer that happens to share the same input
      obj.copied = other ;
      
      % recurse on inputs that were not copied yet and are not shared
      for i = 1:numel(other.inputs)
        if isa(other.inputs{i}, 'Layer') && ...
         ~any(cellfun(@(o) isequal(other.inputs{i}, o), shared))

          if ~isempty(other.inputs{i}.copied)  % reuse same deep copy
            other.inputs{i} = other.inputs{i}.copied ;
          else  % create a new one
            other.inputs{i} = other.inputs{i}.deepCopyRecursive(shared) ;
          end
        end
      end
      
      % repeat for test-mode inputs
      if ~isequal(other.testInputs, 'same')
        for i = 1:numel(other.testInputs)
          if isa(other.testInputs{i}, 'Layer') && ...
           ~any(cellfun(@(o) isequal(other.testInputs{i}, o), shared))

            if ~isempty(other.testInputs{i}.copied)  % reuse same deep copy
              other.testInputs{i} = other.testInputs{i}.copied ;
            else  % create a new one
              other.testInputs{i} = other.testInputs{i}.deepCopyRecursive(shared) ;
            end
          end
        end
      end
    end
  end
  
  methods (Static)
    function autoNames(modifier)
      % LAYER.AUTONAMES()
      % Sets layer names based on the name of the corresponding variables
      % in the caller's workspace.
      %
      % LAYER.AUTONAMES(MODIFIER)
      % Specifies a function handle to be evaluated on each name, possibly
      % modifying it (e.g. append a prefix or suffix).
      %
      % Example:
      %    images = Input() ;
      %    Layer.autoNames() ;
      %    >> images.name
      %    ans =
      %       'images'
      
      if nargin == 0
        modifier = @deal ;
      end
      varNames = evalin('caller','who') ;
      for i = 1:numel(varNames)
        layer = evalin('caller', varNames{i}) ;
        if isa(layer, 'Layer') && isempty(layer.name)
          layer.name = modifier(varNames{i}) ;
        end
      end
    end
  end
end

