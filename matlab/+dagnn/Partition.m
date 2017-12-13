classdef Partition < dagnn.ElementWise
    properties
        partition = {1}
        method = 'max' % only supports 'max'
    end
    
    properties (Transient)
        aux % for 'max', 'aux' is 'argmax'
    end
    
    methods
        function outputs = forward(obj, inputs, params)
            x = cell(1, numel(obj.partition));
            c = cell(1, numel(obj.partition));
            for i = 1:numel(obj.partition)
                [x{i}, c{i}] = max(inputs{1}(:,:,obj.partition{i},:), [], 3);
            end
            outputs{1} = cat(3, x{:});
            obj.aux = cat(3, c{:});
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            y = cell(1, numel(obj.partition));
            inputSize = [size(inputs{1},1) size(inputs{1},2) 0 size(inputs{1},4)];
            labelSize = [size(inputs{1},1) size(inputs{1},2) 1 size(inputs{1},4)];
            numPixelsPerImage = prod(inputSize(1:2));
            numPixels = numPixelsPerImage * inputSize(4);
            n = reshape(0:numPixels-1, labelSize);
            for i = 1:numel(obj.partition)
                inputSize(3) = numel(obj.partition{i});
                y{i} = zeros(inputSize, 'like', inputs{1});
                imageVolume = numPixelsPerImage * inputSize(3);
                offset = 1 + mod(n, numPixelsPerImage) + ...
                    imageVolume * fix(n / numPixelsPerImage);
                c = double(obj.aux(:,:,i,:)); % fix
                ci = offset + numPixelsPerImage * max(c - 1,0);
                y{i}(ci) = derOutputs(:,:,i,:);
            end
            derInputs{1} = cat(3, y{:});
            derParams = {};
        end
        
        function outputSizes = getOutputSizes(obj, inputSizes)
            sz = inputSizes{1};
            sz(3) = numel(obj.partition);
            outputSizes{1} = sz;
        end
        
        function obj = Partition(varargin)
            obj.load(varargin);
        end
    end
end
