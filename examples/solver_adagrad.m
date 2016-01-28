function [weights, g_sqr] = solver_adagrad(weights, g_sqr, grad, opts, lr)
%SOLVER_ADAGRAD
%   Example AdaGrad solver, for use with CNN_TRAIN and CNN_TRAIN_DAG.

if isempty(g_sqr)
  g_sqr = 0 ;
end

g_sqr = g_sqr + grad.^2 ;

weights = weights - lr * grad ./ sqrt(g_sqr + opts.epsilon) ;
