%PDIST vl_nnpdist dagnn wrapper
%  Accepts 2 or 3 inputs, where third input is used as variable
%  'instanceWeights' parameter. Derivatives for the 3rd input are not
%  computed.
classdef PDist < dagnn.Loss
  properties
    p = 2;
  end

  methods
    function outputs = forward(obj, inputs, params)
      switch numel(inputs)
        case 2
          outputs{1} = vl_nnpdist(inputs{1}, inputs{2}, obj.p, [], ...
            obj.opts{:}) ;
        case 3
          outputs{1} = vl_nnpdist(inputs{1}, inputs{2}, obj.p, [], ...
            'instanceWeights', inputs{3}, obj.opts{:}) ;
        otherwise
          error('Invalid number of inputs');
      end
      obj.accumulateAverage(inputs, outputs);
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs = cell(1, numel(inputs));
      switch numel(inputs)
        case 2
          [derInputs{1}, derInputs{2}] = vl_nnpdist(inputs{1}, inputs{2}, ...
            obj.p, derOutputs{1}, obj.opts{:}) ;
        case 3
          [derInputs{1}, derInputs{2}] = vl_nnpdist(inputs{1}, inputs{2}, ...
            obj.p, derOutputs{1}, 'instanceWeights', inputs{3}, obj.opts{:}) ;
        otherwise
          error('Invalid number of inputs');
      end
      derParams = {} ;
    end

    function obj = PDist(varargin)
      obj.load(varargin) ;
      obj.loss = 'pdist';
    end
  end
end
