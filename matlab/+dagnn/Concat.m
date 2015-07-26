classdef Concat < dagnn.ElementWise
  properties
    dim = 3
  end

  methods
    function [outputs, scratch] = forward(self, inputs, params)
      outputs{1} = vl_nnconcat(inputs, self.dim) ;
      scratch = cellfun(@size, inputs, 'UniformOutput', false) ;
    end

    function [derInputs, derParams] = backward(self, inputs, params, derOutpus)
      outputs{1} = vl_nnconcat(inputs, derOutputs{1}, 'inputSizes', scratch{1}) ;
    end

    function obj = Concat(varargin)
      obj.load(varargin) ;
    end

  %   function [outputSizes, transforms] = forwardGeometry(self, inputSizes, paramSizes)
  %     outputSizes{1} = sum(cat(self.dim, inputSizes{:}), self.dim) ;
  %     if self.dim == 3
  %       [~, transforms] = forwardGeometry@nn.elementwise(self, inputSizes, paramSizes) ;
  %       transforms = repmat(transforms,numel(inputSizes),1) ;
  %     else
  %       warning('Not implemented') ;
  %       transforms = repmat({eye(6)},numel(inputSizes),1) ;
  %     end
  %   end
  end
end
