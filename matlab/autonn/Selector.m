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
      assert(index > 1, ...
        'Cannot use a Selector to get the first output of a layer (it is returned by the layer itself).') ;
      
      assert(isempty(input.numOutputs) || index <= input.numOutputs, ...
        sprintf('Attempting to get %ith output of a layer with only %i outputs.', index, input.numOutputs)) ;
      
      obj.enableCycleChecks = false ;
      obj.inputs = {input} ;
      obj.index = index ;
      obj.enableCycleChecks = true ;
    end
  end
end
