function [W, b] = vl_nnlstm_params(d, m, varargin)
%VL_NNLSTM_PARAMS
%   [W, B] = VL_NNLSTM_PARAMS(D, M) initializes parameter matrix W and
%   vector B with appropriate randomized values, for an LSTM with D hidden
%   units and a M-dimensional input.
%   
%   VL_NNLSTM_PARAMS(..., 'option', value, ...) accepts the following
%   options:
%
%   `noise`:: 0.1
%     Magnitude of (uniform) randomized initialization values.
%
%   `forgetBias`:: 1
%     Initial value of forget gate bias; typically starts at 1 to ensure
%     the forget gate is rarely activated during the first learning steps.
%
%   `learningRate`:: 1
%     Learning rate factor. If a vector of 2 elements, they specify the W
%     and B factors, respectively.
%
%   `weightDecay`:: 1
%     Weight decay factor. If a vector of 2 elements, they specify the W
%     and B factors, respectively.

% Copyright (C) 2016 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  opts.noise = 0.1 ;
  opts.forgetBias = 1 ;
  opts.learningRate = 1 ;
  opts.weightDecay = 1 ;
  
  opts = vl_argparse(opts, varargin) ;
  
  % create parameters
  noise = opts.noise ;
  W = Param('value', -noise + 2 * noise * rand(4 * d, d + m, 'single')) ;
  b = Param('value', -noise + 2 * noise * rand(4 * d, 1, 'single'));
  b.value(d+1 : 2*d, :) = opts.forgetBias;

  % set learning rate and weight decay
  W.learningRate = opts.learningRate(1) ;
  b.learningRate = opts.learningRate(max(1,end)) ;

  W.weightDecay = opts.weightDecay(1) ;
  b.weightDecay = opts.weightDecay(max(1,end)) ;

end
