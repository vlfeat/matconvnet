function [inputs, testInputs] = vl_nnbnorm_setup(layer)
%VL_NNBNORM_SETUP
%   Create parameters if needed, and use vl_nnbnorm_autonn wrapper for
%   proper handling of test mode.

  assert(isequal(layer.func, @vl_nnbnorm)) ;
  
  % use wrapper
  layer.func = @vl_nnbnorm_autonn ;
  inputs = layer.inputs ;
  
  % create any parameters if needed (scale, bias and moments).
  assert(numel(inputs) >= 1, 'Must specify at least one input for VL_NNBNORM.') ;
  
  if numel(inputs) < 2
    % create scale param. will be initialized with proper number of
    % channels on first run by the wrapper.
    inputs{2} = Param('value', single(1)) ;  % scalars combine with anything
  end
  
  if numel(inputs) < 3
    % create bias param
    inputs{3} = Param('value', single(0)) ;
  end
  
  pos = find(cellfun(@(a) strcmpi(a, 'moments'), inputs(4:end)), 1) ;
  if isempty(pos)
    % create moments param
    moments = Param('value', single(0)) ;
  else
    % moments param was specified
    moments = inputs{pos + 1} ;
    inputs(pos : pos + 1) = [] ;  % delete it; normal mode doesn't use moments
  end
  
  % in normal mode, pass in moments so its derivatives are expected
  inputs = [inputs(1:3), {'normal', moments}, inputs(4:end)] ;
%   layer.numInputDer = 4 ;
  
  % let the wrapper know when it's in test mode
  testInputs = inputs ;
  testInputs{4} = 'test' ;

end

