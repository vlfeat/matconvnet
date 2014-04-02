function [y,dzdw,dzdb] = gfully(x, w, b, dzdy)

y = bsxfun(@plus, w * x, b) ;

if nargin <= 3, return ; end

% backward
y = (dzdy' * w)' ;
dzdw = dzdy * x' ;
dzdb = sum(dzdy,2) ;
