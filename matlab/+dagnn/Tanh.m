classdef Tanh < dagnn.ElementWise
    
    methods
        function outputs = forward(obj, inputs, params)
            outputs{1} = tanh(inputs{1});
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            derInputs{1} = derOutputs{1}.*(1 - tanh(inputs{1}).^2);
            derParams = {};
        end
    end
end
