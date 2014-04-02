function y = grelu(x,dzdy)

if nargin <= 1
  y = max(x,0) ;
else
  y = dzdy ;
  y(x <= 0) = 0 ;
end
