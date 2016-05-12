function [weights, g_sqr] = solver_adagrad(weights, g_sqr, grad, opts, lr)
%SOLVER_ADAGRAD
%   Example AdaGrad solver, for use with CNN_TRAIN and CNN_TRAIN_DAG.
%
%   Set the initial learning rate for AdaGrad in the options for
%   CNN_TRAIN and CNN_TRAIN_DAG. Note that a learning rate that works for
%   SGD may be inappropriate for AdaGrad; the default is 0.001.
%
%   Solver options: (opts.train.solverOpts)
%
%   `epsilon`:: 1e-10
%      Small additive constant to regularize variance estimate.
%
%   `rho`:: []
%      Moving average window for variance update, between 0 and 1 (larger
%      values result in slower/more stable updating). This has the same
%      effect as RHO for AdaDelta and RMSProp. However, because it is not
%      part of standard AdaGrad, it is disabled by default (RHO is empty).
%
%   A possibly undesirable effect of standard AdaGrad is that the update
%   will monotonically decrease to 0, until training eventually stops. This
%   is because the AdaGrad update is inversely proportional to the total
%   variance of the gradients seen so far.
%   By setting RHO to non-empty, a moving average window of the variance
%   is used instead of the total variance, so the update does not
%   monotonically decrease to 0.

% Copyright (C) 2016 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

if isempty(g_sqr)
  g_sqr = 0 ;
end

if isempty(opts.rho)  % standard, accumulate total variance
  g_sqr = g_sqr + grad.^2 ;
else  % moving average window, similar to RMSProp/AdaDelta
  g_sqr = g_sqr * rho + grad.^2 * (1 - rho) ;
end

weights = weights - lr * grad ./ (sqrt(g_sqr) + opts.epsilon) ;
