function varargout = vl_nnwsum(varargin)
%VL_NNWSUM Summary of this function goes here
%   Detailed explanation goes here

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

