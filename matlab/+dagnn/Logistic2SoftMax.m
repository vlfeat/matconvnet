classdef Logistic2SoftMax < dagnn.ElementWise
    
    methods
        function outputs = forward(obj, inputs, params)
            x = vl_nnsigmoid(inputs{1});
            outputs{1} = cat(3, 1-x, x);
            outputs{2} = (inputs{2}+3)/2;
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            dy = derOutputs{1}(:,:,2,:)-derOutputs(:,:,1,:);
            derInputs = vl_nnsigmoid(inputs{1}, dy);
            derParams = {};
        end
        
        function outputSizes = getOutputSizes(obj, inputSizes)
            sz = inputSizes{1};
            sz(3) = 2;
            outputSizes{1} = sz;
        end
        
        function obj = Logistic2SoftMax(varargin)
            obj.load(varargin);
        end
    end
end
