classdef SegmentationLoss < dagnn.Loss
    
    properties
        % weight = 'instance'
        classWeights = []
    end
    
    methods
        function outputs = forward(obj, inputs, params)
            % inputs: {prediction, label, weight}
            mass = sum(sum(inputs{2} ~= 0, 2), 1) + 1;
            weights = 1./mass;
            if numel(inputs) >= 3 && ~isempty(inputs{3}) % has weights
                weights = bsxfun(@times, inputs{3}, weights);
                opts = horzcat(obj.opts, inputs(4:end));
            else
                opts = horzcat(obj.opts, inputs(3:end));
            end
            outputs{1} = vl_nnloss(inputs{1}, inputs{2}, [], 'loss', obj.loss, ...
                'instanceWeights', weights, 'classWeights', obj.classWeights, opts{:});
            n = obj.numAveraged;
            m = n + size(inputs{1},4);
            obj.average = (n * obj.average + double(gather(outputs{1}))) / m;
            obj.numAveraged = m;
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            % inputs: {prediction, label, weight}
            derInputs = cell(size(inputs));
            mass = sum(sum(inputs{2} ~= 0, 2), 1) + 1;
            weights = 1./mass;
            if numel(inputs) >= 3 && ~isempty(inputs{3}) % has weights
                weights = bsxfun(@times, inputs{3}, weights);
                opts = horzcat(obj.opts, inputs(4:end));
            else
                opts = horzcat(obj.opts, inputs(3:end));
            end
            derInputs{1} = vl_nnloss(inputs{1}, inputs{2}, derOutputs{1}, 'loss', obj.loss, ...
                'instanceWeights', weights, 'classWeights', obj.classWeights, opts{:});
            derParams = {};
        end
        
        function obj = SegmentationLoss(varargin)
            obj.load(varargin);
        end
    end
end
