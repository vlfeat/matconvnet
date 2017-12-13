classdef SegmentationVisualize < dagnn.Layer

    properties
        numClasses = 0;
        order = [1 2 3];
        blendRatio = 0.8;
        colorCode = [];
        index = 1:5;
    end
    
    methods
        function outputs = forward(obj, inputs, params)
            % inputs{1}: predictions, logits or probabilities
            % inputs{2}: ground truth labels
            % inputs{3}: original images
            if strcmp(obj.net.mode, 'normal')
                return
            end
            
            inputs{1} = inputs{1}(:,:,:,obj.index);
            inputs{2} = inputs{2}(:,:,:,obj.index);
            inputs{3} = inputs{3}(:,:,:,obj.index);
            if size(inputs{3},3) == 1
                inputs{3} = repmat(inputs{3}, [1 1 3]);
            end
            n = min(numel(obj.index), size(inputs{1}, 4));
            [~, predictions] = max(inputs{1}, [], 3);
            predictions = gather(predictions);
            labels = gather(inputs{2});
            imageCell = cell(1, n);
            for i = 1:n
                I = inputs{3}(:,:,:,i);
                I = imresize(I, [size(labels,1) size(labels,2)]); %%%###
                L_GT = labels(:,:,:,i);
                L_pred = predictions(:,:,:,i);
                Ic_GT = obj.colorize(I, L_GT);
                Ic_pred = obj.colorize(I, L_pred);
                imageCell{i} = cat(2, Ic_pred, Ic_GT, I);
            end
            figure(100);
            imshow(cat(1, imageCell{:}), []);
            outputs = {};
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            derInputs{1} = [];
            derInputs{2} = [];
            derInputs{3} = [];
            derParams = {};
        end
        
        function Ic = colorize(obj, I, L)
            sz = [size(I,1) size(I,2)];
            Lc = zeros([sz 3]);
            for l = setdiff(unique(L(:))', 0)
                mask = L==l;
                for j = 1:3
                    Lc(:,:,j) = Lc(:,:,j) + mask*obj.colorCode(l,j);
                end
            end
            Ic = (1-obj.blendRatio)*I + obj.blendRatio*cast(Lc,'like',I);
        end
        
        function obj = SegmentationVisualize(varargin)
            obj.load(varargin);
            if isempty(obj.colorCode)
                obj.colorCode = randi(255, [obj.numClasses 3]);
            end
        end
    end
end
