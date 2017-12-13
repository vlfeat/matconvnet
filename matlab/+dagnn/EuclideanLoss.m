classdef EuclideanLoss < dagnn.Loss
    
    methods
        function outputs = forward(obj, inputs, params)
            % inputs: {prediction, groundtruth, <weights>}
            mass = size(inputs{1},1) * size(inputs{1},2);
            if numel(inputs) == 3
                weights = bsxfun(@times, inputs{3}, 1./mass);
            else
                weights = 1./mass;
            end
            outputs{1} = vl_nneuclideanloss(inputs{1}, inputs{2}, [], ...
                'instanceWeights', weights);
            n = obj.numAveraged;
            m = n + size(inputs{1},4);
            obj.average = (n*obj.average + double(gather(outputs{1}))) / m;
            obj.numAveraged = m;
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            mass = size(inputs{1},1) * size(inputs{1},2);
            if numel(inputs) == 3
                weights = bsxfun(@times, inputs{3}, 1./mass);
                derInputs{3} = [];
            else
                weights = 1./mass;
            end
            derInputs{1} = vl_nneuclideanloss(inputs{1}, inputs{2}, derOutputs{1}, ...
                'instanceWeights', weights);
            derInputs{2} = [];
            derParams = {};
        end
        
        function obj = EuclideanLoss(varargin)
            obj.load(varargin);
        end
    end
end
