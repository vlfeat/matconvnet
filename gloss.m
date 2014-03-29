function Y = gloss(X,c,dzdy)

tau = 1e-10 ;
Y = - log(max(X(c),tau)) ;

if nargin <= 2, return ; end

Y = X*0 ;
Y(c) = - 1./X(c) * dzdy ;
