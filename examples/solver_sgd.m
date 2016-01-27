function [weights, momentum] = solver_sgd(weights, momentum, grad, opts, lr)
%SOLVER_SGD
%   Example SGD solver, for use with CNN_TRAIN and CNN_TRAIN_DAG.

if isempty(momentum)
  momentum = 0 ;
end

momentum = opts.momentum * momentum - grad ;
weights = weights + lr * momentum ;
