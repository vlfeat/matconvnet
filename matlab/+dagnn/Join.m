classdef Join < dagnn.ElementWise
    %SUM DagNN sum layer
    %   The SUM layer takes the sum of all its inputs and store the result
    %   as its only output.
    
    properties (Transient)
        numInputs
        activeInputs
    end
    
    methods
        function outputs = forward(obj, inputs, params)
            obj.numInputs = numel(inputs) ;
            
            if strcmp(obj.net.mode, 'test')
                obj.activeInputs = 1 : obj.numInputs;
            else
                obj.activeInputs = [];
                while isempty(obj.activeInputs)
                    obj.activeInputs = find(rand(obj.numInputs,1) > .15);
                end
            end
            %obj.activeInputs = obj.numInputs;%
            
            outputs{1} = inputs{obj.activeInputs(1)} ;
            for k = 2:numel(obj.activeInputs)
                outputs{1} = outputs{1} + inputs{obj.activeInputs(k)} ;
            end
            outputs{1} = outputs{1} / numel(obj.activeInputs);
            
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            for k = 1:obj.numInputs
                if any(obj.activeInputs == k)
                    derInputs{k} = derOutputs{1} / numel(obj.activeInputs);
                else
                    derInputs{k} = derOutputs{1}*0 ;
                end
            end
            
            derParams = {} ;
        end
        
        function outputSizes = getOutputSizes(obj, inputSizes)
            outputSizes{1} = inputSizes{1} ;
            for k = 2:numel(inputSizes)
                if all(~isnan(inputSizes{k})) && all(~isnan(outputSizes{1}))
                    if ~isequal(inputSizes{k}, outputSizes{1})
                        warning('Sum layer: the dimensions of the input variables is not the same.') ;
                    end
                end
            end
        end
        
        function rfs = getReceptiveFields(obj)
            numInputs = numel(obj.net.layers(obj.layerIndex).inputs) ;
            rfs.size = [1 1] ;
            rfs.stride = [1 1] ;
            rfs.offset = [1 1] ;
            rfs = repmat(rfs, numInputs, 1) ;
        end
        
        function obj = Sum(varargin)
            obj.load(varargin) ;
        end
    end
end
