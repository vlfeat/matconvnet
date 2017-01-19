function inputs = vl_nnbnorm_setup(layer)
%VL_NNBNORM_SETUP
%   Create parameters if needed, and use VL_NNBNORM_WRAPPER for proper
%   handling of test mode. Also handles 'learningRate' and 'weightDecay'
%   arguments for the Params, and sets trainMethod of moments to
%   'average'.
%   Called by AUTONN_SETUP.

% Copyright (C) 2016 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  assert(isequal(layer.func, @vl_nnbnorm)) ;
  
  % use wrapper
  layer.func = @vl_nnbnorm_wrapper ;
  inputs = layer.inputs ;
  
  % parse options. note the defaults for bnorm's Params are set here.
  opts = struct('learningRate', [2 1 0.1], 'weightDecay', 0, 'moments', []) ;
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
  
  if ~isempty(opts.moments)
    moments = opts.moments ;
  else
    % 'moments' name-value pair not specified.
    % check if the moments were passed in as the 4th argument (alternative
    % syntax)
    if numel(inputs) > 3 && ~ischar(inputs{4})
      moments = inputs{4} ;
      inputs(4) = [] ;  % remove from list
    else
      % create moments param. note the training method is 'average'.
      moments = Param('value', single(0), ...
                      'learningRate', opts.learningRate(3), ...
                      'weightDecay', opts.weightDecay(3), ...
                      'trainMethod', 'average') ;
    end
  end
  
  % in normal mode, pass in moments so its derivatives are expected.
  % create Input('testMode') to know when in test mode.
  inputs = [inputs(1:3), {moments, Input('testMode')}, inputs(4:end)] ;
  layer.numInputDer = 4 ;
  
end

