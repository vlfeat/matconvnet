classdef OhemSoftMaxLabel < dagnn.ElementWise
    % OhemSoftMaxLabel
    
    properties
        minKept = 0 % per image
        maxKept = Inf
        topK = 2
        margin = -1
        threshold = 0.6
        classFrequencies = []
    end
    
    methods
        function outputs = forward(obj, inputs, params)
            % inputs: {prediction, label}
            % outputs: {label}
            opts = horzcat({}, inputs(3:end));
            outputs{1} = vl_nnohemsoftmaxlabel(inputs{1}, inputs{2}, ...
                'threshold', obj.threshold, 'topK', obj.topK, ...
                'margin', obj.margin, 'minKept', obj.minKept, 'maxKept', obj.maxKept, ...
                'classFrequencies', obj.classFrequencies, opts{:});
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            derInputs = cell(size(inputs));
            derParams = {};
        end
        
        function outputSizes = getOutputSizes(obj, inputSizes)
            % inputs: {prediction, label}
            outputSizes = inputSizes(2);
        end
        
        function obj = OhemSoftMaxLabel(varargin)
            obj.load(varargin);
        end
    end
end
