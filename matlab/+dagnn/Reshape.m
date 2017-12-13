classdef Reshape < dagnn.Layer
    properties
        inputSize = [0 0 0 0]
        outputSize = [0 0 0 0]
    end
    
    methods
        function outputs = forward(obj, inputs, params)
            outputs{1} = vl_nnreshape(inputs{1}, obj.outputSize);
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            derInputs{1} = vl_nnreshape(derOutputs{1}, obj.inputSize);
            derParams = {};
        end
        
        function outputSizes = getOutputSizes(obj, inputSizes)
            outputSizes{1} = [obj.outputSize NaN];
        end
        
        function obj = Reshape(varargin)
            obj.load(varargin);
        end
    end
end
