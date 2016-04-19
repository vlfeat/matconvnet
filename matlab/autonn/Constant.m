classdef Constant < Param
%Constant
%   Defines an explicit constant in the network. This differs from implicit
%   constants, which are part of function calls in autonn, in three ways:
%   1. Its value can be moved to and from the GPU using NET.MOVE.
%   2. Its value can be set after network construction, with NET.SETVALUE.
%   3. Its derivative can be retrieved with NET.GETDER.
%
%   Internally, this is just a Param with trainMethod set to 'none'.

% Copyright (C) 2016 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  methods
    function obj = Constant(value, varargin)
      obj@Param('value',value, 'trainMethod','none', ...
        'weightDecay',0, 'learningRate',0)
    end
  end
end
