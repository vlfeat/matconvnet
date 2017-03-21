function y = gpuArray_wrapper(x, gpuMode, dzdy)
%GPUARRAY_WRAPPER
%   Wrapper for gpuArray, disabled automatically when in CPU mode.

  if nargin < 3  % forward mode
    if gpuMode
      y = gpuArray(x) ;
    else
      y = x ;
    end
  else  % backward mode
    if ~isa(x, 'gpuArray')
      y = gather(dzdy) ;  % convert derivative to same type as input
    else
      y = dzdy ;  % keep same type (was already a gpuArray)
    end
  end
end
