function [Y,dzdW] = gfully(X, W, dzdY)

Y = W * X ;

if nargin <= 2, return ; end

% backwad
Y = dzdY'*W ;
dzdW = dzdY * X' ;
