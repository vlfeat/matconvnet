function [y, db] = vl_nnmatrixop(a, b, op, dy)
%VL_NNMATRIXOP Summary of this function goes here
%   Detailed explanation goes here

  if nargin < 4
    % forward function. cannot use singleton expansion for matrix ops
    if isempty(b)
      y = op(a) ;
    else
      y = op(a, b) ;
    end
  else
    % backward function
    if isequal(op, @transpose)
      da = dy.' ;
      
    elseif isequal(op, @ctranspose)
      da = dy' ;
      
    elseif isequal(op, @mtimes)
      da = dy * b.' ;
      db = a.' * dy ;
      
    elseif isequal(op, @mrdivide)
      % note: @mldivide is just @mrdivide with swapped inputs (see vl_nnmatrixop_setup)
      da = dy / b ;
      db = [] ;
      error('Not implemented.') ;
      
    else
      error('Derivative not implemented.') ;
    end
    
    y = da ;
  end
end

