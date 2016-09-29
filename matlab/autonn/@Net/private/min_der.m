function [da, db] = min_der(a, b, dim, dy)
%MIN_DER
%   MIN_DER(A, DY)
%   MIN_DER(A, B, DY)
%   MIN_DER(A, [], DIM, DY)
%   Derivative of MIN function. Same syntax as native MIN, plus derivative.

% Copyright (C) 2016 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  if nargin == 4
    % derivative of MIN(A, [], DIM)
    assert(isempty(b))
    
  elseif nargin == 3
    % derivative of MIN(A, B)
    dy = dim ;  % 3rd argument
    if isscalar(dy)  % if dy is scalar, replicate it to size of A or B
      dy = dy + zeros(max(size(a), size(b))) ;
    end
    i = (a <= b);
    
    if isscalar(a)
      da = sum(dy(i)) ;
    else
      da = zeros(size(a), 'like', a) ;
      da(i) = dy(i) ;
    end
    
    if isscalar(b)
      db = sum(dy(~i)) ;
    else
      db = zeros(size(b), 'like', b) ;
      db(~i) = dy(~i) ;
    end
    
    return
    
  elseif nargin == 2
    % derivative of MIN(A)
    dy = b;  % 2nd argument
    dim = find([size(a), 2] ~= 1, 1) ;  % find first non-singleton dim
  end
  
  % permute so that dim is 1 (otherwise the masked assignment later on
  % won't index the correct elements, for 1 < dim <= ndims(a))
  perm = [dim, 1:dim-1, dim+1:ndims(a)] ;
  a = permute(a, perm) ;
  
  % min elements along dim have a derivative (the corresponding element
  % from dy), others are 0
  [~, i] = min(a, [], 1) ;
  
  % select min elements
  mask = bsxfun(@eq, (1:size(a,1)).', i) ;
  
  da = zeros(size(a), 'like', a) ;
  da(mask) = dy ;
  
  da = ipermute(da, perm) ;

end

