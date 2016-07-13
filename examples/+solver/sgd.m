function [w, momentum] = sgd(w, momentum, grad, opts, lr)
%SGD
%   Example SGD solver, with momentum, for use with CNN_TRAIN and
%   CNN_TRAIN_DAG.
%
%   The convergence of SGD depends heavily on the learning rate (set in the
%   options for CNN_TRAIN and CNN_TRAIN_DAG).
%
%   If called without any input argument, returns the default options
%   structure.
%
%   Solver options: (opts.train.solverOpts)
%
%   `momentum`:: 0.9
%      Parameter for Momentum SGD; set to 0 for standard SGD.
%
%   Note: for backwards compatibility, the parameter can also be set in
%   opts.train.momentum.

% Copyright (C) 2016 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

if nargin == 0 % Return the default solver options
  w = struct('momentum', 0.9);
  return;
end
if isempty(momentum)
  momentum = 0 ;
end

momentum = opts.momentum * momentum - grad ;
w = w + lr * momentum ;
