function [weights, g_sqr] = solver_rmsprop(weights, g_sqr, grad, opts, lr)
%SOLVER_RMSPROP
%   Example RMSProp solver, for use with CNN_TRAIN and CNN_TRAIN_DAG.

if isempty(g_sqr)
  g_sqr = 0 ;
end

g_sqr = g_sqr * opts.rho + grad.^2 * (1 - opts.rho) ;

weights = weights - lr * grad ./ sqrt(g_sqr + opts.epsilon) ;
