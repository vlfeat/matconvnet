function [weights, g_sqr] = solver_adagrad(weights, g_sqr, grad, opts, lr)
%SOLVER_ADAGRAD
%   Example AdaGrad solver, for use with CNN_TRAIN and CNN_TRAIN_DAG.
%
%   Set the initial learning rate for AdaGrad in the options for
%   CNN_TRAIN and CNN_TRAIN_DAG. Note the default learning rate is probably
%   inappropriate for AdaGrad; values around 0.001 may work better.
%
%   Solver options: (opts.train.solverOpts)
%
%   `epsilon`:: 1e-10
%      Small additive constant to regularize variance estimate.

% Copyright (C) 2016 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

if isempty(g_sqr)
  g_sqr = 0 ;
end

g_sqr = g_sqr + grad.^2 ;

weights = weights - lr * grad ./ (sqrt(g_sqr) + opts.epsilon) ;
