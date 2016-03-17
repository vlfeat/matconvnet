function [y, db] = vl_nnbinaryop(a, b, op, dy)
%VL_NNBINARYOP Summary of this function goes here
%   Detailed explanation goes here

  if nargin < 4
    % forward function. use singleton expansion (important when generalizing
    % a function to work with minibatches).
    y = bsxfun(op, a, b) ;
  else
    % backward function
    if isequal(op, @times)
      da = bsxfun(@times, b, dy) ;
      db = bsxfun(@times, a, dy) ;

    elseif isequal(op, @rdivide)
      % note: @ldivide is just @rdivide with swapped inputs (see Layer)
      da = bsxfun(@rdivide, dy, b) ;
      db = -dy .* bsxfun(@rdivide, a, b .^ 2) ;

    elseif isequal(op, @power)
      da = dy .* a .^ (b - 1) .* b ;
      db = dy .* (a .^ b) .* log(a) ;

    else
      error('Derivative not implemented.') ;
    end

    % now sum derivatives along any expanded dimensions (by bsxfun)
    for t = 1:ndims(dy)  % ndims(dy) is an upper bound on ndims of a and b
      if size(a,t) == 1  % this means the original was a singleton dimension
        da = sum(da, t) ;
      end
      if size(b,t) == 1
        db = sum(db, t) ;
      end
    end
    
    y = da ;
  end
end

