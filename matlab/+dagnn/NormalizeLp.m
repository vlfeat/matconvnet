classdef NormalizeLp < dagnn.ElementWise
    properties
        p = 2
        epsilon = 1e-4
        spatial = false
    end

    methods
        function outputs = forward(obj, inputs, params)
            outputs{1} = vl_nnnormalizelp(inputs{1}, [], ...
                'p', obj.p, 'epsilon', obj.epsilon, 'spatial', obj.spatial);
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            derInputs{1} = vl_nnnormalizelp(inputs{1}, derOutputs{1}, ...
                'p', obj.p, 'epsilon', obj.epsilon, 'spatial', obj.spatial);
            derParams = {};
        end

        function obj = NormalizeLp(varargin)
            obj.load(varargin{:});
        end
    end
end
