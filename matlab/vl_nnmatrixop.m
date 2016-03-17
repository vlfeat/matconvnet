function [y, db] = vl_nnmatrixop(a, b, op, dy)
%VL_NNMATRIXOP Summary of this function goes here
%   Detailed explanation goes here

  if nargin < 4
    % forward function. cannot use singleton expansion for matrix ops
    y = op(a, b) ;
  else
    % backward function
    if isequal(op, @mtimes)
      da = dy * b.' ;
      db = a.' * dy ;
      
    elseif isequal(op, @mrdivide)
      % note: @mldivide is just @mrdivide with swapped inputs (see constructor)
      da = dy / b ;
      db = [] ;
      error('Not implemented.') ;
      
    else
      error('Derivative not implemented.') ;
    end
    
    y = da ;
  end
end

