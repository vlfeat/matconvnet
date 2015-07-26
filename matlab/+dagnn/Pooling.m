classdef Pooling < dagnn.Filter
  properties
    method = 'max'
    poolSize = [1 1]
    opts = {'cuDNN'}
  end

  methods
    function outputs = forward(self, inputs, params)
      outputs{1} = vl_nnpool(inputs{1}, self.poolSize, ...
                             'pad', self.pad, ...
                             'stride', self.stride, ...
                             'method', self.method, ...
                             self.opts{:}) ;
    end

    function [derInputs, derParams] = backward(self, inputs, params, derOutputs)
      derInputs{1} = vl_nnpool(inputs{1}, self.poolSize, derOutputs{1}, ...
                               'pad', self.pad, ...
                               'stride', self.stride, ...
                               'method', self.method, ...
                               self.opts{:}) ;
      derParams = {} ;
    end

    function obj = Pooling(varargin)
      obj.load(varargin) ;
    end

    % function [outputSizes, transforms] = forwardGeometry(self, inputSizes, paramSizes)
    %   transforms{1} = [...
    %     self.stride(1), 0, 1 - self.pad(1) - self.stride(1) 0 0 0 ;
    %     0, self.stride(2), 1 - self.pad(3) - self.stride(2) 0 0 0 ;
    %     0, 0, 1, 0, 0, 0, ;
    %     0, 0, 0, self.stride(1), 0, 1 - self.pad(1) - self.stride(1) + self.poolSize(1) - 1 ;
    %     0, 0, 0, 0, self.stride(2), 1 - self.pad(3) - self.stride(2) + self.poolSize(2) - 1 ;
    %     0, 0, 0, 0, 0, 1] ;
    %   outputSizes{1}(1:2) = floor((inputSizes{1}(1:2) + self.pad([1 3]) + self.pad([2 4]) - self.poolSize(:)') ./ self.stride(:)') + 1 ;
    %   outputSizes{1}(3:4) = inputSizes{1}(3:4) ;
    % end
  end
end
