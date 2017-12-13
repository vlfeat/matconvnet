function derParams = vl_gradclip(derParams, varargin)
%VL_NNGRADCLIP Gradient clipping
%   DERPARAMS = VL_NNGRADCLIP(DERPARAMS) where DERPARAMS is a cell list of 
%   derivative of parameters.

% clamp: rescale according to grad norm
% clip : trim by thresholding
opts.epsilon = 1e-4;
opts.method = 'clip';
opts.threshold = Inf;
opts = vl_argparse(opts, varargin, 'nonrecursive');

if isinf(opts.threshold) || isnan(opts.threshold), return; end
switch opts.method
    case 'clip'
        for i = 1:numel(derParams)
            derParams{i} = min(derParams{i},  opts.threshold);
            derParams{i} = max(derParams{i}, -opts.threshold);
        end
    case 'clamp'
        for i = 1:numel(derParams)
            grad_norm = (norm(derParams{i}(:),2)+opts.epsilon)/numel(derParams{i});
            if grad_norm > opts.threshold
                derParams{i} = derParams{i} / grad_norm * opts.threshold;
            end
        end
end
