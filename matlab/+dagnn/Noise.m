classdef Noise < dagnn.ElementWise
    properties
        noiseStd = 0.1;
    end
    
    properties (Transient)
        
    end
    
    methods
        function outputs = forward(obj, inputs, params)
            outputs = inputs;
            for i = 1 : numel(outputs)
                n = randn(size(outputs{i}),'single') * single(obj.noiseStd);
                if isa(outputs{i},'gpuArray')
                    n = gpuArray(n);
                end
                outputs{i} = outputs{i} + n;
            end
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            derInputs = derOutputs;
            derParams = {};
        end
        
        function obj = Noise(varargin)
            obj.load(varargin{:}) ;
        end
    end
end
