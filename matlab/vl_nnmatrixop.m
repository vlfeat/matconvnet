function [y, db] = vl_nnmatrixop(a, b, op, dy)
%VL_NNMATRIXOP
%   Y = VL_NNMATRIXOP(A, [], OP)
%   Matrix unary operation OP (@transpose, @ctranspose).
%
%   Y = VL_NNMATRIXOP(A, B, OP)
%   Matrix binary operation OP (@mtimes, @mrdivide, @mpower).
%
%   DA = VL_NNMATRIXOP(A, [], OP, DY)
%   [DA, DB] = VL_NNMATRIXOP(A, B, OP, DY)
%   Projected derivatives of the same operation, with respect to inputs A
%   and B.

% Copyright (C) 2016 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

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
      % note: @mldivide is just @mrdivide with swapped inputs (see
      % autonn_setup/vl_nnmatrixop_setup)
      da = dy / b ;
      db = [] ;
      error('Not implemented.') ;
      
    elseif isequal(op, @mpower)
      error('Not implemented.') ;
      
    else
      error('Derivative not implemented.') ;
    end
    
    y = da ;
  end
end

