function derFunc = autonn_der(func)
%AUTONN_DER
%   AUTONN_DER is only called by Net during construction.
%
%   Given a function handle, returns the function handle for its
%   derivative. It has the same name as the function, followed by '_der'.
%
%   Small derivative functions are defined as subfunctions here.

% Copyright (C) 2016 Joao Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  derFunc = str2func([func2str(func) '_der']) ;
  info = functions(derFunc) ;
  
  % if '<func>_der' is undefined, '<func>' itself must implement the
  % derivative
  if isempty(info.file)
    derFunc = func ;
  end
end

function dx = reshape_der(x, varargin)  %#ok<*DEFNU>
  dx = reshape(varargin{end}, size(x)) ;
end

function dx = permute_der(~, dim, dy)
  dx = ipermute(dy, dim) ;
end

function dx = ipermute_der(~, dim, dy)
  dx = permute(dy, dim) ;
end

function dx = sqrt_der(x, dy)
  dx = dy ./ sqrt(x) ;
end

function dx = sum_der(x, dim, dy)
  if nargin < 3
    % one-argument syntax of sum, plus derivative
    dy = dim;
    dim = find([size(x), 2] ~= 1, 1) ;  % find first non-singleton dim
  end

  % repeat dy along the summed dimension
  reps = ones(1, ndims(x)) ;
  reps(dim) = size(x,dim) ;
  dx = repmat(dy, reps) ;
end

function dx = mean_der(x, dim, dy)
  if nargin < 3
    % one-argument syntax of mean, plus derivative
    dy = dim;
    dim = find([size(x), 2] ~= 1, 1) ;  % find first non-singleton dim
  end

  % repeat dy along the summed dimension
  reps = ones(1, ndims(x)) ;
  reps(dim) = size(x,dim) ;
  dx = repmat(dy, reps) / size(x, dim) ;
end

function size_der(~, ~, ~)
  % nothing to return; SIZE_SETUP (in AUTONN_SETUP) specifies 0 derivatives
end

function varargout = root_der(varargin)
  % copy the output derivative to all input derivatives (see ROOT).
  varargout = cell(1, numel(varargin) - 1) ;
  varargout(:) = varargin(end) ;
end

