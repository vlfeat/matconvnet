classdef SegmentationAccuracy < dagnn.Loss
    
    properties (Transient)
        pixelAccuracy = 0
        meanAccuracy = 0
        meanIntersectionUnion = 0
        confusion = 0
    end
    
    properties
        numClasses = 0
        metrics = {'pixelAccuracy', 'meanAccuracy', 'meanIntersectionUnion'}
        accumulateStats = true
    end
    
    methods
        function outputs = forward(obj, inputs, params)
            % inputs: {score, label}
            [~, predictions] = max(inputs{1}, [], 3);
            predictions = gather(predictions);
            labels = gather(inputs{2});
            
            % compute statistics only on accumulated pixels when
            % obj.accumulateStats is true, otherwise on current batch
            assert(size(labels,3) == 1);
            ok = labels > 0;
            numPixels = nnz(ok);
            
            if obj.accumulateStats
                numAveraged = obj.numAveraged + numPixels;
                obj.confusion = obj.confusion + ...
                    accumarray([labels(ok) predictions(ok)], 1, ...
                    [obj.numClasses obj.numClasses]);
                
                % compute various statistics of the confusion matrix
                rel = sum(obj.confusion, 2);  % denominator of recall/acc, relevant elements
                sel = sum(obj.confusion, 1)'; % denominator of precision, selected elements
                tp  = diag(obj.confusion);   % true positive
                
                obj.pixelAccuracy = sum(tp) / max(1, numAveraged);
                obj.meanAccuracy = mean(tp ./ max(1, rel));
                obj.meanIntersectionUnion = mean(tp ./ max(1, rel + sel - tp));
            else
                obj.confusion = accumarray([labels(ok) predictions(ok)], 1, ...
                    [obj.numClasses obj.numClasses]);
                
                % compute various statistics of the confusion matrix
                rel = sum(obj.confusion, 2);  % denominator of recall/acc, relevant elements
                sel = sum(obj.confusion, 1)'; % denominator of precision, selected elements
                tp  = diag(obj.confusion);   % true positive
                
                acc = sum(tp) / max(1, numPixels);
                mAcc = mean(tp ./ max(1, rel));
                mIU = mean(tp ./ max(1, rel + sel - tp));
                
                n = obj.numAveraged;
                m = n + numPixels;
                obj.pixelAccuracy = (n*obj.pixelAccuracy + numPixels*acc) / m;
                obj.meanAccuracy = (n*obj.meanAccuracy + numPixels*mAcc) / m;
                obj.meanIntersectionUnion = (n*obj.meanIntersectionUnion + numPixels*mIU) / m;
            end
            obj.numAveraged = obj.numAveraged + numPixels;
            
            % Only outputs averaged metrics
            [obj.average, outputs] = deal(zeros(1,numel(obj.metrics)), cell(1,numel(obj.metrics)));
            for i = 1:numel(obj.metrics)
                [obj.average(i), outputs{i}] = deal(obj.(obj.metrics{i}));
            end
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            derInputs{1} = [];
            derInputs{2} = [];
            derParams = {};
        end
        
        function reset(obj)
            obj.confusion = 0;
            obj.pixelAccuracy = 0;
            obj.meanAccuracy = 0;
            obj.meanIntersectionUnion = 0;
            obj.average = [0 0 0]';
            obj.numAveraged = 0;
        end
        
        function str = toString(obj)
            str = sprintf('acc:%.2f%%, mAcc:%.2f%%, mIU:%.2f%%', ...
                obj.pixelAccuracy*100, ...
                obj.meanAccuracy*100, ...
                obj.meanIntersectionUnion*100);
        end

        function obj = SegmentationAccuracy(varargin)
            obj.load(varargin);
        end
    end
end
