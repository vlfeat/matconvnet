function Y = gloss(X,c,dzdy)

% no division by zero
X = X + 1e-4 ;
c_ = c+(0:size(X,3):size(X,3)*size(X,4)-1) ;
  
if nargin <= 2
  Y = - sum(sum(sum(log(X(:,:,c_))))) ;
else
  Y_ = - (1./X) * dzdy ;
  Y = Y_*0 ;
  Y(:,:,c_) = Y_(:,:,c_) ;
end
