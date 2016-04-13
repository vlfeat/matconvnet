function [y, dzdg, dzdb, moments] = vl_nnbnorm_autonn(x, g, b, moments, test, varargin)
%VL_NNBNORM_AUTONN
%   VL_NNBNORM has a non-standard interface (returns a derivative for the
%   moments, even though they are not an input), so we must wrap it.
%   VL_NNBNORM_SETUP replaces a standard VL_NNBNORM call with this one.
%
%   This also lets us supports nice features like setting the parameter
%   sizes automatically (e.g. building a net with VL_NNBNORM(X) is valid).

% Copyright (C) 2016 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  % use number of channels in X to extend scalar (i.e. default) params to
  % the correct size
  if size(x, 3) > 1
    if isscalar(g)
      g(1:size(x,3),1) = g ;
    end
    
    if isscalar(b)
      b(1:size(x,3),1) = b ;
    end
    
    if numel(moments) == 2
      moments = repmat(moments(:)', size(x,3), 1) ;
    end
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

