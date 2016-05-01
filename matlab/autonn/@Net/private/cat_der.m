function varargout = cat_der(dim, varargin)
%CAT_DER
%   CAT_DER(DIM, X1, X2, ..., DZDY)
%   Derivative of CAT function. Same syntax as native CAT, plus derivative.
%   Note the first output is always empty (derivative of DIM).
%
%   This must be different from VL_NNCONCAT, because we need to take a
%   variable number of inputs (for Matlab syntax compatibility), and the
%   backward function must be separate from the forward function in order
%   to avoid ambiguity in the last argument (is it a derivative or not?).
%   It also works with arbitrary numbers of dimensions.
  
% Copyright (C) 2016 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  dzdy = varargin{end} ;
  
  if isscalar(dzdy)
    % special case, a scalar derivative propagates to all non-empty inputs
    valid = ~cellfun('isempty', varargin(1:end-1)) ;
    valid = [false, valid] ;  % add slot for DIM derivative, which is 0
    varargout = cell(size(valid)) ;
    varargout(valid) = {dzdy} ;
    varargout{1} = 0 ;
    return
  end

  % create indexing structure
  idx = cell(1, ndims(dzdy)) ;
  idx(:) = {':'} ;
  
  start = 1 ;
  varargout = cell(1, numel(varargin)) ;
  
  for i = 1 : numel(varargin) - 1
    sz = size(varargin{i}, dim) ;
    idx{dim} = start : start + sz - 1 ;
    varargout{i + 1} = dzdy(idx{:}) ;
    start = start + sz ;
  end
  varargout{1} = 0 ;  % DIM derivative

end

