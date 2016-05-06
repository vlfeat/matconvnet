function [weights, state] = solver_adadelta(weights, state, grad, opts, ~)
%SOLVER_ADADELTA
%   Example AdaDelta solver, for use with CNN_TRAIN and CNN_TRAIN_DAG.
%
%   Solver options: (opts.train.solverOpts)
%
%   `epsilon`:: 1e-8
%      Small additive constant to regularize variance estimate.
%
%   `rho`:: 0.95
%      Moving average window for variance update (larger values result in
%      slower/more stable updating).

% Copyright (C) 2016 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

if isempty(state)
  state.g_sqr = 0 ;
  state.delta_sqr = 0 ;
end

rho = opts.rho ;

state.g_sqr = state.g_sqr * rho + grad.^2 * (1 - rho) ;
new_delta = -sqrt((state.delta_sqr + opts.epsilon) ./ ...
                  (state.g_sqr + opts.epsilon)) .* grad ;
state.delta_sqr = state.delta_sqr * rho + new_delta.^2 * (1 - rho) ;

weights = weights + new_delta ;
