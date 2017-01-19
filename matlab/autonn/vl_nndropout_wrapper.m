function y = vl_nndropout_wrapper(x, mask, test, dzdy)
%VL_NNDROPOUT_WRAPPER

% Copyright (C) 2017 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  if test
    % pass-through input unchanged, when in test-mode
    if nargin < 4
      y = x ;
    else  % test-mode backward pass
      y = dzdy ;
    end
  else
    if nargin < 4
      y = vl_nndropout(x, 'mask', mask) ;
    else  % backward pass
      y = vl_nndropout(x, dzdy, 'mask', mask) ;
    end
  end
end

