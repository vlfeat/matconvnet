classdef RandIndexMetric < dagnn.Loss
    
    properties (Transient)
        randIndex = 0
        randIndexAfterThinning = 0
        randInfo = 0
        randInfoAfterThinning = 0
    end
    
    properties
        metrics = {'randIndex', 'randIndexAfterThinning', 'randInfo', 'randInfoAfterThinning'}
        computeRandIndex = true
        computeRandInfo = false
    end
    
    methods
        function outputs = forward(obj, inputs, params)
            % inputs: {prediction, groundtruth}
            % 2 is foreground, 1 is background, ignore 0
            [~, predictions] = max(inputs{1}, [], 3);
            predictions = gather(predictions);
            labels = gather(inputs{2});
            labels = labels + (labels == 0); % set null label to background
            predictions = logical(predictions-1);
            labels = logical(labels-1);
            
            [thisIndex, thisInfo] = vl_nnrandindex(predictions, labels, ...
                'thin', 'none', 'computeRandIndex', obj.computeRandIndex, ...
                'computeRandInfo', obj.computeRandInfo);
            [thisIndexThin, thisInfoThin] = vl_nnrandindex(predictions, labels, ...
                'thin', 'watershed', 'computeRandIndex', obj.computeRandIndex, ...
                'computeRandInfo', obj.computeRandInfo);
            
            n = obj.numAveraged;
            m = n + size(inputs{1},4);
            obj.numAveraged = m;
            
            obj.randIndex = (n*obj.randIndex + sum(thisIndex)) / m;
            obj.randIndexAfterThinning = (n*obj.randIndexAfterThinning + sum(thisIndexThin)) / m;
            obj.randInfo = (n*obj.randInfo + sum(thisInfo)) / m;
            obj.randInfoAfterThinning = (n*obj.randInfoAfterThinning + sum(thisInfoThin)) / m;
            
            % Only outputs averaged metrics
            [obj.average, outputs] = deal(zeros(1,numel(obj.metrics)), cell(1,numel(obj.metrics)));
            for i = 1:numel(obj.metrics)
                [obj.average(i), outputs{i}] = deal(obj.(obj.metrics{i}));
            end
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            derInputs{1} = [];
            derInputs{2} = [];
            derParams = {};
        end
        
        function reset(obj)
            obj.randIndex = 0;
            obj.randIndexAfterThinning = 0;
            obj.randInfo = 0;
            obj.randInfoAfterThinning = 0;
            obj.average = [0 0 0 0];
            obj.numAveraged = 0;
        end
        
        function obj = RandIndexMetric(varargin)
            obj.load(varargin);
        end
    end
end
