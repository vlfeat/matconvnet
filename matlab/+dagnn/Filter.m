classdef Filter < dagnn.Layer
  properties
    pad = [0 0 0 0]
    stride = [1 1]
  end
  methods
    function [outputSizes, transforms] = forwardGeometry(self, inputSizes, paramSizes)
      outputSizes = {} ;
      transforms = {} ;
    end

  function set.pad(obj, pad)
      if numel(pad) == 1
        obj.pad = [pad pad pad pad] ;
      elseif numel(pad) == 2
        obj.pad = pad([1 1 2 2]) ;
      else
        obj.pad = pad ;
      end
    end

    function set.stride(obj, stride)
      if numel(stride) == 1
        obj.stride = [stride stride] ;
      else
        obj.stride = stride ;
      end
    end
  end
end
