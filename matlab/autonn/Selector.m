classdef Selector < Layer
%Selector
%   Used to implement layers with multiple outputs.
%   This layer simply returns the Nth output of its input layer, where N
%   is the INDEX property.

% Copyright (C) 2016 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  properties
    index
  end
  
  methods
    function obj = Selector(input, index)
      obj.inputs = {input} ;
      obj.index = index ;
    end
  end
end
