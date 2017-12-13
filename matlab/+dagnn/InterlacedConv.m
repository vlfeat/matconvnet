classdef InterlacedConv < dagnn.Filter
    % INTERLACEDCONV Interlaced convolution
    
    properties
        size = [0 0 0 0]
        hasBias = true
        reshape = false
        upsample = 1
        opts = {'cuDNN'}
    end
    
    methods
        function outputs = forward(obj, inputs, params)
            y = cell(1, obj.upsample^2);
            c = obj.size(4)/obj.upsample^2;
            for i = 1:obj.upsample^2
                if obj.hasBias, b = params{2}((i-1)*c+1:i*c); else, b = []; end
                y{i} = vl_nnconv(...
                    inputs{1}, params{1}(:,:,:,(i-1)*c+1:i*c), b, ...
                    'pad', obj.pad, ...
                    'stride', obj.stride, ...
                    'dilate', obj.dilate, ...
                    obj.opts{:});
            end
            if ~obj.reshape
                outputs{1} = cat(3, y{:});
            else
                sizeY = [size(y{1}) 1 1 1 1]; %#ok
                sizeY = [obj.upsample*sizeY(1:2) sizeY(3:4)];
                outputs{1} = zeros(sizeY, 'like', y{1});
                for i = 1:obj.upsample^2
                    [offsetY, offsetX] = ind2sub([1 1]*obj.upsample, i);
                    outputs{1}(offsetY:obj.upsample:end,offsetX:obj.upsample:end,:,:) = y{i};
                end
            end
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            derInputs{1} = 0;
            df = cell(1, obj.upsample^2);
            db = cell(1, obj.upsample^2);
            c = obj.size(4)/obj.upsample^2;
            for i = 1:obj.upsample^2
                if ~obj.reshape
                    dy = derOutputs{1}(:,:,(i-1)*c+1:i*c,:);
                else
                    [offsetY, offsetX] = ind2sub([1 1]*obj.upsample, i);
                    dy = derOutputs{1}(offsetY:obj.upsample:end,offsetX:obj.upsample:end,:,:);
                end
                if obj.hasBias, b = params{2}((i-1)*c+1:i*c); else, b = []; end
                [dx, df{i}, db{i}] = vl_nnconv(...
                    inputs{1}, params{1}(:,:,:,(i-1)*c+1:i*c), b, dy, ...
                    'pad', obj.pad, ...
                    'stride', obj.stride, ...
                    'dilate', obj.dilate, ...
                    obj.opts{:});
                derInputs{1} = derInputs{1} + dx;
            end
            derParams{1} = cat(4, df{:});
            derParams{2} = cat(1, db{:});
            derParams = vl_gradclip(derParams, obj.clip{:});
        end
        
        function kernelSize = getKernelSize(obj)
            kernelSize = obj.size(1:2);
        end
        
        function outputSizes = getOutputSizes(obj, inputSizes)
            outputSizes = getOutputSizes@dagnn.Filter(obj, inputSizes);
            outputSizes{1}(3) = obj.size(4);
            if obj.reshape
                outputSizes{1}(1:2) = obj.upsample*outputSizes{1}(1:2);
                outputSizes{1}(3) = outputSizes{1}(3)/obj.upsample^2;
            end
        end
        
        function params = initParams(obj)
            % Xavier improved
            sc = sqrt(2/prod(obj.size(1:3)));
            params{1} = randn(obj.size, 'single')*sc;
            if obj.hasBias
                params{2} = zeros(obj.size(4), 1, 'single');
            end
        end
        
        function set.size(obj, ksize)
            % make sure that ksize has 4 dimensions
            ksize = [ksize(:)' 1 1 1 1];
            obj.size = ksize(1:4);
        end
        
        function obj = InterlacedConv(varargin)
            obj.load(varargin);
            % normalize field by implicitly calling setters defined in
            % dagnn.Filter and here
            obj.size = obj.size;
            obj.stride = obj.stride;
            obj.pad = obj.pad;
            obj.clip = obj.clip;
        end
    end
end
