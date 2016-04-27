classdef Slice < dagnn.ElementWise
    properties
        dim = 3
        pts = [];
    end
    
    properties (Transient)
        outputSizes = {}
    end
    
    methods
        function outputs = forward(obj, inputs, params)
            sz = repmat(size(inputs{1}),[length(obj.pts)+1,1]);
            szd = diff([0,obj.pts-1,size(inputs{1},obj.dim)]);
            sz(:,obj.dim) = szd;
            obj.outputSizes = num2cell(sz,2);
            outputs = vl_nnconcat(inputs, obj.dim, inputs{1}, 'inputSizes', obj.outputSizes) ;
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            derInputs{1} = vl_nnconcat(derOutputs, obj.dim) ;
            derParams = {} ;
        end
        
        function reset(obj)
            obj.outputSizes = {} ;
        end
        
        function outputSizes = getOutputSizes(obj, inputSizes)
            sz = repmat(inputSizes{1},[length(obj.pts)+1,1]);
            szd = diff([0,obj.pts-1,inputSizes{1}(obj.dim)]);
            sz(:,obj.dim) = szd;
            outputSizes = num2cell(sz,2);
        end
        
        function rfs = getReceptiveFields(obj)
            numInputs = numel(obj.net.layers(obj.layerIndex).inputs) ;
            if obj.dim == 3 || obj.dim == 4
                rfs = getReceptiveFields@dagnn.ElementWise(obj) ;
                rfs = repmat(rfs, numInputs, 1) ;
            else
                for i = 1:numInputs
                    rfs(i,1).size = [NaN NaN] ;
                    rfs(i,1).stride = [NaN NaN] ;
                    rfs(i,1).offset = [NaN NaN] ;
                end
            end
        end
        
        function load(obj, varargin)
            s = dagnn.Layer.argsToStruct(varargin{:}) ;
            % backward file compatibility
            if isfield(s, 'numInputs'), s = rmfield(s, 'numInputs') ; end
            load@dagnn.Layer(obj, s) ;
        end
        
        function obj = Slice(varargin)
            obj.load(varargin{:}) ;
        end
    end
end
