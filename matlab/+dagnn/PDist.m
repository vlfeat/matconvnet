%PDIST vl_nnpdist dagnn wrapper
%  Accepts 2 or 3 inputs, where third input is used as variable
%  'instanceWeights' parameter. Derivatives for the 3rd input are not
%  computed.
%  By default aggregates the element-wise loss.
classdef PDist < dagnn.Loss
  properties
    p = 2;
    aggregate = true;
    noRoot = false;
    epsilon = 1e-6;
  end

  methods
    function outputs = forward(obj, inputs, params)
      args = {'aggregate', obj.aggregate, 'noRoot', obj.noRoot, ...
          'epsilon', obj.epsilon};
      switch numel(inputs)
        case 2
          outputs{1} = vl_nnpdist(inputs{1}, inputs{2}, obj.p, [], ...
              args{:}, obj.opts{:}) ;
        case 3
          outputs{1} = vl_nnpdist(inputs{1}, inputs{2}, obj.p, [], ...
            args{:}, 'instanceWeights', inputs{3}, obj.opts{:}) ;
        otherwise
          error('Invalid number of inputs');
      end
      obj.accumulateAverage(inputs, outputs);
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs = cell(1, numel(inputs));
      args = {'aggregate', obj.aggregate, 'noRoot', obj.noRoot, ...
          'epsilon', obj.epsilon};
      switch numel(inputs)
        case 2
          [derInputs{1}, derInputs{2}] = vl_nnpdist(inputs{1}, inputs{2}, ...
            obj.p, derOutputs{1}, args{:}, obj.opts{:}) ;
        case 3
          [derInputs{1}, derInputs{2}] = vl_nnpdist(inputs{1}, inputs{2}, ...
            obj.p, derOutputs{1}, args{:}, 'instanceWeights', inputs{3}, ...
            obj.opts{:}) ;
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
