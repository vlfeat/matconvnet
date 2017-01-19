function inputs = vl_nndropout_setup(layer)
%VL_NNDROPOUT_SETUP
%   Setup a dropout layer, by adding a mask generator layer as input and
%   wrapping it.
%   Called by AUTONN_SETUP.

% Copyright (C) 2016 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  assert(isequal(layer.func, @vl_nndropout)) ;
  
  inputs = layer.inputs ;
  x = inputs{1} ;

  % parse dropout rate
  rate = 0.5 ;
  if numel(inputs) >= 3 && strcmp(inputs{2}, 'rate')
    rate = inputs{3} ;
  end
  
  % create mask generator
  maskLayer = Layer(@vl_nnmask, x, rate) ;
  
  % replace function signature with the wrapper:
  %   y = vl_nndropout_wrapper(x, mask, test)
  layer.func = @vl_nndropout_wrapper ;
  inputs = {x, maskLayer, Input('testMode')} ;
  
  % vl_nndropout doesn't return a derivative for the mask
  layer.numInputDer = 1 ;

end

