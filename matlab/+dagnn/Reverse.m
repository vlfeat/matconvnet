classdef Reverse < dagnn.Layer
    
    properties
        mode = 'reverse'
    end
    
    methods
        function outputs = forward(obj, inputs, params)
            outputs{1} = inputs{1};
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            if numel(inputs) > 1 && ~isempty(inputs{2})
                obj.mode = inputs{2};
            end
            switch lower(obj.mode)
                case 'reverse'
                    derInputs{1} = -derOutputs{1};
                case 'identity'
                    derInputs{1} = derOutputs{1};
                case 'zero'
                    derInputs{1} = 0*derOutputs{1};
            end
            derInputs{2} = [];
            derParams = {};
        end
        
        function outputSizes = getOutputSizes(obj, inputSizes)
            outputSizes{1} = inputSizes{1};
        end
        
        function obj = Reverse(varargin)
            obj.load(varargin);
        end
    end
end
