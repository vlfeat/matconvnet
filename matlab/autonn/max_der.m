function [da, db] = max_der(a, b, dim, dy)
%MAX_DER
%   MAX(A, DY)
%   MAX(A, B, DY)
%   MAX(A, [], DIM, DY)
%   Derivative of MAX function. Same syntax as native MAX, plus derivative.

  if nargin == 4
    % derivative of MAX(A, [], DIM)
    assert(isempty(b))
    
  elseif nargin == 3
    % derivative of MAX(A, B)
    dy = dim;  % 3rd argument
    i = (a > b);
    
    da = zeros(size(a), 'like', a);
    da(i) = dy(i);
    
    db = zeros(size(b), 'like', b);
    db(~i) = dy(~i);
    
    return
    
  elseif nargin == 2
    % derivative of MAX(A)
    dy = b;  % 2nd argument
    dim = find([size(a), 2] ~= 1, 1) ;  % find first non-singleton dim
  end
  
  % max elements along dim have a derivative (the corresponding element
  % from dy), others are 0
  [~, i] = max(a, [], dim) ;
  
  % indexes 1:size(a,dim), pushed to dimension dim
  idx = reshape(1:size(a,dim), [ones(1, dim-1), size(a,dim), 1, 1]) ;

  % select max elements
  mask = bsxfun(@eq, idx, i) ;
  
  da = zeros(size(a), 'like', a) ;
  da(mask) = dy(mask) ;

end

