function y = vl_nnnative(varargin)
%VL_NNNATIVE
%   Implementation of differentiable versions of most native Matlab
%   functions (not operators). Call with the string name of a native
%   function to get a function handle to its differentiable version.

  %return handle to the appropriate local function
  if nargin == 1 && ischar(varargin{1})
    y = str2func(['nn' varargin{1}]);
  end
end

function y = nnsum(x, dim, dy)
  if nargin < 3  % forward
    y = sum(x, dim) ;
  else  % backward
    reps = ones(1, ndims(x)) ;
    reps(dim) = size(x,dim) ;
    y = repmat(dy, reps) ;  %repeat dy along the summed dimension
  end
end

function y = nnmean(x, dim, dy)
  if nargin < 3  % forward
    y = sum(x, dim) / size(x,dim) ;
  else  % backward
    reps = ones(1, ndims(x)) ;
    reps(dim) = size(x,dim) ;
    y = repmat(dy, reps) / size(x,dim) ;  %repeat dy along the summed dim.
  end
end

function y = nnreshape(x, sz, dy)  %#ok<*DEFNU>
  if nargin < 3  % forward
      if iscell(sz)  % allows an empty dimension size, computed automatically
        y = reshape(x, sz{:}) ;
      else
        y = reshape(x, sz) ;
      end
  else  % backward
    y = reshape(dy, size(x)) ;
  end
end
