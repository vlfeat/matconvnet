function [inputs, testInputs] = vl_nnbnorm_setup(layer)
%VL_NNBNORM_SETUP
%   Create parameters if needed, and use VL_NNBNORM_AUTONN wrapper for
%   proper handling of test mode. Also handles 'learningRate' and
%   'weightDecay' arguments for the Params.
%   Called by AUTONN_SETUP.

% Copyright (C) 2016 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  assert(isequal(layer.func, @vl_nnbnorm)) ;
  
  % use wrapper
  layer.func = @vl_nnbnorm_autonn ;
  inputs = layer.inputs ;
  
  % parse options
  opts = struct('learningRate', 1, 'weightDecay', 1) ;
  [opts, inputs] = vl_argparsepos(opts, inputs) ;
  
  if isscalar(opts.learningRate)
    opts.learningRate = opts.learningRate([1 1 1]) ;
  end
  if isscalar(opts.weightDecay)
    opts.weightDecay = opts.weightDecay([1 1 1]) ;
  end
  
  % create any unspecified parameters (scale, bias and moments).
  assert(numel(inputs) >= 1, 'Must specify at least one input for VL_NNBNORM.') ;
  
  if numel(inputs) < 2
    % create scale param. will be initialized with proper number of
    % channels on first run by the wrapper.
    inputs{2} = Param('value', single(1), ...
                      'learningRate', opts.learningRate(1), ...
                      'weightDecay', opts.weightDecay(1)) ;
  end
  
  if numel(inputs) < 3
    % create bias param
    inputs{3} = Param('value', single(0), ...
                      'learningRate', opts.learningRate(2), ...
                      'weightDecay', opts.weightDecay(2)) ;
  end
  
  pos = find(cellfun(@(a) strcmpi(a, 'moments'), inputs(4:end)), 1) ;
  if isempty(pos)
    % create moments param
    moments = Param('value', single(0), ...
                    'learningRate', opts.learningRate(3), ...
                    'weightDecay', opts.weightDecay(3)) ;
  else
    % moments param was specified
    moments = inputs{pos + 1} ;
    inputs(pos : pos + 1) = [] ;  % delete it; normal mode doesn't use moments
  end
  
  % in normal mode, pass in moments so its derivatives are expected
  inputs = [inputs(1:3), {moments, false}, inputs(4:end)] ;
  layer.numInputDer = 4 ;
  
  % let the wrapper know when it's in test mode
  testInputs = inputs ;
  testInputs{5} = true ;

end

