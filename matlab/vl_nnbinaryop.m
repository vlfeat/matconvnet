function [y, db] = vl_nnbinaryop(a, b, op, dy)
%VL_NNBINARYOP
%   Y = VL_NNBINARYOP(A, B, OP)
%   Arithmetic binary operation OP (@times, @rdivide, @power), with
%   binary singleton expansion enabled (BSXFUN).
%
%   [DA, DB] = VL_NNBINARYOP(A, B, OP, DY)
%   Projected derivatives of the same operation, with respect to inputs A
%   and B.

% Copyright (C) 2016 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  if nargin < 4
    % forward function. use singleton expansion (important when generalizing
    % a function to work with minibatches).
    y = bsxfun(op, a, b) ;
  else
    % backward function
    if isequal(op, @times)
      da = bsxfun(@times, b, dy) ;
      if nargout > 1
        db = bsxfun(@times, a, dy) ;
      end

    elseif isequal(op, @rdivide)
      % note: @ldivide is just @rdivide with swapped inputs (see
      % autonn_setup/vl_nnbinaryop_setup)
      da = bsxfun(@rdivide, dy, b) ;
      if nargout > 1
        db = -dy .* bsxfun(@rdivide, a, b .^ 2) ;
      end

    elseif isequal(op, @power)
      da = dy .* a .^ (b - 1) .* b ;
      if nargout > 1  % prevents error if log(a) becomes complex, but is not needed anyway because b is constant
        db = dy .* (a .^ b) .* log(a) ;
      end

    else
      error('Derivative not implemented.') ;
    end

    % now sum derivatives along any expanded dimensions (by bsxfun)
    for t = 1:ndims(dy)  % ndims(dy) is an upper bound on ndims of a and b
      if size(a,t) == 1  % this means the original was a singleton dimension
        da = sum(da, t) ;
      end
      if nargout > 1 && size(b,t) == 1
        db = sum(db, t) ;
      end
    end
    
    y = da ;
  end
end

