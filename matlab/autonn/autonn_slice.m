function y = autonn_slice(x, varargin)
% Helper function implementing forward indexing operation. The derivative
% is implemented in Net.eval() for efficiency with sparse updates, so
% this does not correspond to the usual vl_nn* interface.

% Copyright (C) 2016 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  y = x(varargin{:}) ;
end

