function varargout = vl_nnwsum(varargin)
%VL_NNWSUM
%   Y = VL_NNWSUM(A, B, ..., 'weights', W)
%   Weighted sum of inputs. Each element of vector W denotes the weight of
%   the corresponding input.
%
%   [DA, DB, ...] = VL_NNWSUM(A, B, ..., DZDY, 'weights', W)
%   Projected derivatives of the same operation with respect to all inputs,
%   except for weights W, which are assumed constant. For products of
%   non-constant inputs, see VL_NNBINARYOP.

% Copyright (C) 2016 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  assert(numel(varargin) >= 2 && isequal(varargin{end-1}, 'weights'), ...
    'Must supply the ''weights'' property.') ;

  w = varargin{end} ;  % vector of scalar weights
  n = numel(varargin) - 2 ;
  
  if n == numel(w)
      % forward function
      y = w(1) * varargin{1} ;
      for k = 2:n
        y = bsxfun(@plus, y, w(k) * varargin{k}) ;
      end
      
      varargout = {y} ;
      
  elseif n == numel(w) + 1
      % backward function (the last argument is the derivative)
      dy = varargin{n} ;
      n = n - 1 ;
      
      varargout = cell(1, n) ;
      for k = 1:n
        dx = dy ;
        for t = 1:ndims(dy)  % sum derivatives along expanded dimensions (by bsxfun)
          if size(varargin{k},t) == 1  % original was a singleton dimension
            dx = sum(dx, t) ;
          end
        end
        varargout{k} = w(k) * dx ;
      end
      
  else
    error('The number of weights does not match the number of inputs.') ;
  end
end

