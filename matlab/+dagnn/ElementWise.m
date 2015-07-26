classdef ElementWise < dagnn.Layer
  methods
    function [outputSizes, transforms] = forwardGeometry(self, inputSizes, paramSizes)
      outputSizes = inputSizes ;
      transforms = {eye(6)} ;
    end
  end
end
