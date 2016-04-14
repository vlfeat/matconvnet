function [da, db] = min_der(a, b, dim, dy)
%MIN_DER
%   MIN(A, DY)
%   MIN(A, B, DY)
%   MIN(A, [], DIM, DY)
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
    dy = dim;  % 3rd argument
    i = (a <= b);
    
    da = zeros(size(a), 'like', a);
    da(i) = dy(i);
    
    db = zeros(size(b), 'like', b);
    db(~i) = dy(~i);
    
    return
    
  elseif nargin == 2
    % derivative of MIN(A)
    dy = b;  % 2nd argument
    dim = find([size(a), 2] ~= 1, 1) ;  % find first non-singleton dim
  end
  
  % min elements along dim have a derivative (the corresponding element
  % from dy), others are 0
  [~, i] = min(a, [], dim) ;
  
  % indexes 1:size(a,dim), pushed to dimension dim
  idx = reshape(1:size(a,dim), [ones(1, dim-1), size(a,dim), 1, 1]) ;

  % select min elements
  mask = bsxfun(@eq, idx, i) ;
  
  da = zeros(size(a), 'like', a) ;
  da(mask) = dy(mask) ;

end

