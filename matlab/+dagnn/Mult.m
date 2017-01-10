classdef Mult < dagnn.ElementWise
    %MULT DagNN mult layer
    %   The MULT layer takes the multiplication of all its inputs and store the result
    %   as its only output.
    
    properties (Transient)
        numInputs
    end
    
    methods
        function outputs = forward(obj, inputs, params)
            obj.numInputs = numel(inputs) ;
            outputs{1} = inputs{1} ;
            for k = 2:obj.numInputs
                outputs{1} = outputs{1} .* inputs{k} ;
            end
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            for k = 1:obj.numInputs
                derInputs{k} = derOutputs{1} ;
                for j = setdiff(1:obj.numInputs,k)
                    derInputs{k} = derInputs{k} .* inputs{j};
                end
            end
            derParams = {} ;
        end
        
        function outputSizes = getOutputSizes(obj, inputSizes)
            outputSizes{1} = inputSizes{1} ;
            for k = 2:numel(inputSizes)
                if all(~isnan(inputSizes{k})) && all(~isnan(outputSizes{1}))
                    if ~isequal(inputSizes{k}, outputSizes{1})
                        warning('Mult layer: the dimensions of the input variables is not the same.') ;
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
        
        function obj = Mult(varargin)
            obj.load(varargin) ;
        end
    end
end
