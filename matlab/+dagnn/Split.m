classdef Split < dagnn.ElementWise
    properties
        childIds = {};
    end
    
    methods
        function outputs = forward(obj, inputs, params)
            assert(numel(inputs)==1);
            for i = 1:numel(obj.childIds)
                outputs{i} = inputs{1}(:,:,obj.childIds{i},:);
            end
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            derInput = zeros(size(inputs{1}), 'like', inputs{1});
            for i = 1:numel(obj.childIds)
                derInput(:,:,obj.childIds{i},:) = ...
                    derInput(:,:,obj.childIds{i},:) + derOutputs{i};
            end
            derInputs{1} = derInput;
            derParams{1} = [];
        end
        
        function outputSizes = getOutputSizes(obj, inputSizes)
            isz = inputSizes{1};
            for i = 1:numel(obj.childIds)
                outputSizes{i} = [isz(1),isz(2),numel(obj.childIds{i}),isz(4)];
            end
        end
        
        function rfs = getReceptiveFields(obj)
            numInputs = numel(obj.net.layers(obj.layerIndex).inputs);
            numOutputs = numel(obj.childIds);
            rfs.size = [1 1];
            rfs.stride = [1 1];
            rfs.offset = [1 1];
            rfs = repmat(rfs, numInputs, numOutputs);
        end
        
        function obj = Split(varargin)
            obj.load(varargin);
        end
    end
end
