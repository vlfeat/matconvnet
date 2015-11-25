classdef BatchNorm < dagnn.ElementWise
  methods
    function outputs = forward(obj, inputs, params)
      outputs{1} = vl_nnbnorm(inputs{1}, params{1}, params{2}) ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs{1} = vl_nnrelu(inputs{1}, derOutputs{1}) ;
      [derInputs{1}, derParams{1}, derParams{2}] = ...
          vl_nnbnorm(inputs{1}, params{1}, params{2}, derOutputs{1}) ;
    end
  end
end
