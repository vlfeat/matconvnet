classdef ElementWise < dagnn.Layer
  methods
    function [outputSizes, transforms] = forwardGeometry(self, inputSizes, paramSizes)
      outputSizes = inputSizes ;
      transforms = {eye(6)} ;
    end
    
    function transformations = getSpatialTransformations(obj)
      transformations = { [1 0 1 ; 1 0 1] } ;
    end
    
    function outputSizes = getOutputSizes(obj, inputSizes, paramSizes)
      outputSizes = inputSizes ;
    end  
  end
end
