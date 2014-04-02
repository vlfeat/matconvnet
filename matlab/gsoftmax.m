function Y = gsoftmax(X,dzdY)

E = exp(bsxfun(@minus, X, max(X,[],1))) ;
L = sum(E) ;
Y = bsxfun(@rdivide, E, L) ;

if nargin <= 1, return ; end

% backward
Y = Y .* bsxfun(@minus, dzdY, sum(dzdY .* Y,1)) ;
