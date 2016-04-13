classdef Input < Layer
%Input
%   Defines a network input (such as images or labels).

% Copyright (C) 2016 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  methods
    function obj = Input(name)
      if nargin >= 1
        obj.name = name ;
      end
    end
  end
end
