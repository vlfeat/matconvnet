function [y, dzdg, dzdb, moments] = vl_nnbnorm_wrapper(x, g, b, moments, test, varargin)
%VL_NNBNORM_WRAPPER
%   VL_NNBNORM has a non-standard interface (returns a derivative for the
%   moments, even though they are not an input), so we must wrap it.
%   VL_NNBNORM_SETUP replaces a standard VL_NNBNORM call with this one.
%
%   This also lets us supports nice features like setting the parameter
%   sizes automatically (e.g. building a net with VL_NNBNORM(X) is valid).
%
%   Note that MOMENTS is pre-multiplied by the number of images in the
%   batch (similar to dagnn.BatchNorm, and unlike VL_NNBNORM).

% Copyright (C) 2016 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  % use number of channels in X to extend scalar (i.e. default) params to
  % the correct size. this way the layer can be constructed without
  % knowledge of the number of channels. scalars also permit gradient
  % accumulation with any tensor shape (in CNN_TRAIN_AUTONN).
  if isscalar(g)
    g(1:size(x,3),1) = g ;
  end
  if isscalar(b)
    b(1:size(x,3),1) = b ;
  end
  if isscalar(moments)
    moments(1:size(x,3),1:2) = moments;
  end

  if test
    % test mode, pass in moments
    y = vl_nnbnorm(x, g, b, 'moments', moments, varargin{:}) ;
  else
    if numel(varargin) >= 1 && isnumeric(varargin{1})
      % backward mode
      [y, dzdg, dzdb, moments] = vl_nnbnorm(x, g, b, varargin{:}) ;
    else
      % forward
      y = vl_nnbnorm(x, g, b, varargin{:}) ;
    end
  end

end

