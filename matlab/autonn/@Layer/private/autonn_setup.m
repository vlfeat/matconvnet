function inputs = autonn_setup(obj)
%AUTONN_SETUP
%   AUTONN_SETUP is only called by Layer during construction.
%
%   Defines special behavior for a Layer, by calling the setup function
%   associated with its function handle. A setup function, if it exists,
%   has the same name as the original function, followed by '_setup'.
%
%   Small setup functions are defined as subfunctions here.

% Copyright (C) 2016 Joao Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  % do not modify inputs by default
  inputs = obj.inputs ;
  
  % several native functions have a very simple setup: just specify that
  % their inputs are not differentiable.
  non_differentiable = {'size', 'colon', 'rand', 'randn', 'randi', ...
    'randperm', 'ones', 'zeros', 'inf', 'nan', 'eye', ...
    'eq', 'ne', 'lt', 'gt', 'le', 'ge', 'and', 'or', 'not', 'xor', ...
    'any', 'all'} ;
  if any(strcmp(func2str(obj.func), non_differentiable))
    obj.numInputDer = 0 ;
  else
    % check existence of setup function
    setupFunc = str2func([func2str(obj.func) '_setup']) ;
    info = functions(setupFunc) ;

    if ~isempty(info.file)
      % optional return value (an updated inputs list)
      out = cell(1, nargout(setupFunc)) ;
      [out{:}] = setupFunc(obj) ;

      if numel(out) >= 1  % changed inputs
        inputs = out{1} ;
      end
    end
  end
end

function vl_nnloss_setup(layer)
  % the 2nd input (label) has no derivative.
  layer.numInputDer = 1 ;
end

function vl_nnsoftmaxloss_setup(layer)
  % the 2nd input (label) has no derivative.
  layer.numInputDer = 1 ;
end

function inputs = vl_nnconvt_setup(layer)
  % same setup as for conv layers (see VL_NNCONV_SETUP).
  inputs = vl_nnconv_setup(layer) ;
end

function inputs = vl_nnbinaryop_setup(layer)  %#ok<*DEFNU>
  % @ldivide is just @rdivide with swapped inputs.
  inputs = layer.inputs ;
  if isequal(inputs{3}, @ldivide)
    inputs = [inputs([2,1]), {@rdivide}] ;
  end
end

function inputs = vl_nnmatrixop_setup(layer)
  % @mldivide is just @mrdivide with swapped inputs.
  inputs = layer.inputs ;
  if isequal(inputs{3}, @mldivide)
    inputs = [inputs([2,1]), {@mrdivide}] ;
  end
end

function repmat_setup(layer)
  % only first derivative defined for REPMAT
  layer.numInputDer = 1 ;
end

function reshape_setup(layer)
  % only first derivative defined for RESHAPE
  layer.numInputDer = 1 ;
end

