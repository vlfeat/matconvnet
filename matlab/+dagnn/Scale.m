classdef Scale < dagnn.ElementWise
  properties
    size
    hasBias = true
  end

  methods
    function outputs = forward(obj, inputs, params)

      if numel(inputs) >= 2, params{1} = inputs{2} ; end
      outputs{1} = bsxfun(@times, inputs{1}, params{1}) ;

      if obj.hasBias
        if numel(inputs) >= 2, params{2} = inputs{3} ; end
        outputs{1} = bsxfun(@plus, outputs{1}, params{2}) ;
      end
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)

      % The scale/bias can be passed in as either input or parameter
      if numel(inputs) >= 2, params{1} = inputs{2} ; end
      derInputs{1} = bsxfun(@times, derOutputs{1}, params{1}) ;
      derParams{1} = derOutputs{1} .* inputs{1} ;
      for k = find(size(params{1}) == 1)
        if size(inputs{1},k) > 1
          derParams{1} = sum(derParams{1},k) ;
        end
      end

      if obj.hasBias
        if numel(inputs) >= 2, params{2} = inputs{3} ; end
        derParams{2} = derOutputs{1} ;
        for k = find(size(params{2}) == 1)
          if size(inputs{1},k) > 1
            derParams{2} = sum(derParams{2},k) ;
          end
        end
      end

      if numel(inputs) == 1
        derInputs(2:3) = derParams ;
        derParams = {} ;
      end
    end

    function obj = Scale(varargin)
      obj.load(varargin) ;
    end
  end
end
