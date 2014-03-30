function [y,dzdw] = gfully(x, w, dzdy)

y = w * x ;

if nargin <= 2, return ; end

% backward
y = (dzdy' * w)' ;
dzdw = dzdy * x' ;
