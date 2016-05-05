function dx = repmat_der(~, varargin)
%REPMAT_DER
%   REPMAT_DER(X, N, DZDY)
%   REPMAT_DER(X, D1, D2, ..., DZDY)
%   REPMAT_DER(X, D, DZDY)
%   Derivative of REPMAT function, w.r.t. first input. Same syntax as
%   native REPMAT, plus derivative.

% Copyright (C) 2016 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  dzdy = varargin{end} ;
  
  assert(numel(varargin) >= 2, 'Invalid syntax.') ;
  
  % collect repetitions for each dimension in a single vector
  if numel(varargin) == 2
    reps = varargin{1} ;
    if isscalar(reps)
      reps = [reps, reps] ;
    end
  else
    reps = [varargin{1:end-1}] ;
  end
  
  % iterate dimensions, summing derivative accross all repetitions in that
  % dimension
  dx = dzdy ;
  for dim = 1:numel(reps)
    if reps(dim) > 1
      % split dimension in 2: one for the original, one for the repetitions
      sz = size(dx) ;
      dx = reshape(dx, [sz(1:dim-1), sz(dim) / reps(dim), reps(dim), sz(dim+1:end)]) ;
      
      % sum across the repetitions dimension
      dx = sum(dx, dim + 1) ;
      
      % merge the two dimensions again
      dx = reshape(dx, [sz(1:dim-1), sz(dim) / reps(dim), sz(dim+1:end)]) ;
    end
  end
end

