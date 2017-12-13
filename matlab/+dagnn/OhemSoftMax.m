classdef OhemSoftMax < dagnn.Loss
    
    properties
        minKept = 0 % per image
        maxKept = Inf
        topK = 2
        margin = -1
        threshold = 0.6
        classFrequencies = []
        classWeights = []
    end
    
    methods
        function outputs = forward(obj, inputs, params)
            % inputs: {prediction, label, weight}
            % forward pass: weights is normalized by number of pixels
            % (mass).
            mass = sum(sum(inputs{2} ~= 0, 2), 1) + 1;
            weights = 1./mass;
            if numel(inputs) >= 3 && ~isempty(inputs{3}) % has weights
                weights = bsxfun(@times, inputs{3}, weights);
                opts = horzcat(obj.opts, inputs(4:end));
            else
                opts = horzcat(obj.opts, inputs(3:end));
            end
            outputs{1} = vl_nnohemsoftmax(inputs{1}, inputs{2}, [], ...
                'minKept', obj.minKept, 'maxKept', obj.maxKept, ...
                'topK', obj.topK, 'margin', obj.margin, 'threshold', obj.threshold, ...
                'classFrequencies', obj.classFrequencies, 'classWeights', obj.classWeights, ...
                'instanceWeights', weights, opts{:});
            n = obj.numAveraged;
            m = n + size(inputs{1},4);
            obj.average = (n * obj.average + double(gather(outputs{1}))) / m;
            obj.numAveraged = m;
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            % inputs: {prediction, label, weight}
            % backward pass: weights is not normalized by number of pixels,
            % since valid number of pixels is not known before-hand.
            derInputs = cell(size(inputs));
            batchSize = size(inputs{2}, 4);
            weights = batchSize;
            if numel(inputs) >= 3 && ~isempty(inputs{3}) % has weights
                weights = bsxfun(@times, inputs{3}, weights);
                opts = horzcat(obj.opts, inputs(4:end));
            else
                opts = horzcat(obj.opts, inputs(3:end));
            end
            derInputs{1} = vl_nnohemsoftmax(inputs{1}, inputs{2}, derOutputs{1}, ...
                'minKept', obj.minKept, 'maxKept', obj.maxKept, ...
                'topK', obj.topK, 'margin', obj.margin, 'threshold', obj.threshold, ...
                'classFrequencies', obj.classFrequencies, 'classWeights', obj.classWeights, ...
                'instanceWeights', weights, opts{:});
            derParams = {};
        end
        
        function obj = OhemSoftMax(varargin)
            obj.load(varargin);
        end
    end
end
