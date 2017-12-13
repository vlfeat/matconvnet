% inputs: {top-down, bottom-up}
classdef UnPooling < dagnn.Filter
    properties
        method = 'max'
        poolSize = [1 1]
        opts = {'cuDNN'}
        argmax = []
    end
    
    methods
        function outputs = forward(obj, inputs, params)
            % compute argmax from inputs{2} (using backprop of pooling)
            onemap = ones(size(inputs{1}),'single');
            if isa(inputs{1}, 'gpuArray') || isa(inputs{2}, 'gpuArray')
                onemap = gpuArray(onemap);
            end
            obj.argmax = vl_nnpool(inputs{2}, obj.poolSize, onemap, ...
                'pad', obj.pad, 'stride', obj.stride, ...
                'method', obj.method, obj.opts{:});
            % apply argmax to upsample inputs{1}
            outputs{1} = vl_nnpool(obj.argmax, obj.poolSize, inputs{1}, ...
                'pad', obj.pad, 'stride', obj.stride, ...
                'method', obj.method);
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            % use argmax as mask for average pooling
            mask = obj.argmax * prod(obj.poolSize); % to cancel the average
            derInputs{1} = vl_nnpool(derOutputs{1}.*mask, obj.poolSize, ...
                'pad', obj.pad, 'stride', obj.stride, ...
                'method', 'avg', obj.opts{:});
            derInputs{2} = [];
            derParams = {};
        end
        
        function kernelSize = getKernelSize(obj)
            kernelSize = obj.poolSize;
        end
        
        function outputSizes = getOutputSizes(obj, inputSizes)
            assert(all(obj.pad==0) && all(obj.poolSize==2));
            outputSizes{1} = inputSizes{1};
            outputSizes{1}(1:2) = outputSizes{1}(1:2) .* obj.stride(1:2);
        end
        
        function obj = UnPooling(varargin)
            obj.load(varargin);
        end
    end
end
