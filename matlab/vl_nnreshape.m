function y = vl_nnreshape(x, sz, dy)
%VL_NNRESHAPE Summary of this function goes here
%   Detailed explanation goes here
  
  if nargin < 3  % forward
      if iscell(sz)  % allows an empty dimension size, computed automatically
        y = reshape(x, sz{:}) ;
      else
        y = reshape(x, sz) ;
      end
  else  % backward
    y = reshape(dy, size(x)) ;
  end
end
