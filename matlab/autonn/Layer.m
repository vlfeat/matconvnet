classdef Layer < handle

  properties (SetAccess = private, GetAccess = public)  % not setable after construction, to ensure a proper call graph (no cycles)
    inputs = {}  % list of inputs, either constants or other Layers
    testInputs = 'same'  % list of inputs used in test mode, may be different
  end
  
  properties
    func = []  % main function being called
    testFunc = []  % function called in test mode (empty to use the same as in normal mode; 'none' to disable, e.g. dropout)
    name = []  % optional name (for debugging mostly; a layer is a unique handle object that can be passed around)
    numInputDer = []  % to manually specify the number of input derivatives returned in bwd mode
  end
  
  properties (SetAccess = {?Net}, GetAccess = public)
    idx = 0  % index of this layer in the Net, used only during its construction
  end
  
  methods
    function obj = Layer(func, varargin)  % wrap a function call
      if isa(obj, 'Input') || isa(obj, 'Param')
        return  % these do not need a function call
      end
      
      % general case
      obj.func = func ;
      obj.inputs = varargin(:)' ;
      
      % call setup function if needed. it can change the inputs list (not
      % allowed for outside functions, to preserve call graph structure).
      setup_func = [func2str(func) '_setup'] ;
      if exist(setup_func, 'file')
        setup_func = str2func(setup_func) ;
        
        % accept between 0 and 2 return values
        out = cell(1, nargout(setup_func)) ;
        [out{:}] = setup_func(obj) ;
        
        if numel(out) >= 1  % changed inputs
          obj.inputs = out{1} ;
        end
        
        if numel(out) >= 2  % changed test mode inputs
          obj.testInputs = out{2} ;
        end
      end
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
      c = Layer(@vl_nnmatrixop, a, b, @mtimes) ;
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
      fprintf('\n%s = ', name) ;
      
      if ~isempty(name) && ~isempty(obj.inputs)
        fprintf('%s(', char(obj.func)) ;
        
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
      end
      fprintf('\n\n') ;
      
      disp(obj) ;
    end
  end
  
  methods (Access = {?Net, ?Layer})
    function resetOrder(obj)
      obj.idx = 0 ;
      for i = 1:numel(obj.inputs)  % recurse on inputs
        if isa(obj.inputs{i}, 'Layer')
          obj.inputs{i}.resetOrder() ;
        end
      end
    end
    
    function layers = buildOrder(obj, layers)
      % recurse on inputs with unassigned indexes (not on the list yet)
      for i = 1:numel(obj.inputs)
        if isa(obj.inputs{i}, 'Layer') && obj.inputs{i}.idx == 0
          layers = obj.inputs{i}.buildOrder(layers) ;
        end
      end
      
      % add self to the execution order, after all inputs are there
      layers{end+1} = obj ;
      obj.idx = numel(layers) ;
    end
  end
  
  methods (Static)
    function autoNames()
      % set layer names based on the name of the corresponding variables in
      % the caller's workspace (e.g. if x=vl_nnconv(...), then x.name='x').
      varNames = evalin('caller','who') ;
      for i = 1:numel(varNames)
        layer = evalin('caller', varNames{i}) ;
        if isa(layer, 'Layer') && isempty(layer.name)
          layer.name = varNames{i} ;
        end
      end
    end
  end
end

