function Y = gsoftmax(X,dzdY)

E = exp(X) ;
L = sum(E) ;
Y = E / L ;
  
if nargin <= 1, return ; end

% backward
Y = Y .* (dzdY - sum(dzdY .* Y)) ;
