function y = vl_nneuclideanloss(x, c, varargin)
%EUCLIDEANLOSS

if ~isempty(varargin) && ~ischar(varargin{1})  % passed in dzdy
    dzdy = varargin{1};
    varargin(1) = [];
else
    dzdy = [];
end

opts.instanceWeights = [];
opts = vl_argparse(opts, varargin, 'nonrecursive');

assert(numel(x) == numel(c));

if ~isempty(opts.instanceWeights)
    % important: this code needs to broadcast opts.instanceWeights to an 
    % array of the same size as c
    instanceWeights = bsxfun(@times, onesLike(c), opts.instanceWeights);
end

if nargin <= 2 || isempty(dzdy)
    t = 1/2 * (x - c).^2;
    if ~isempty(instanceWeights)
        y = instanceWeights(:)' * t(:);
    else
        y = sum(t(:));
    end
    y = gather(y);
else
    if ~isempty(instanceWeights)
        dzdy = dzdy * instanceWeights;
    end
    y = dzdy .* (x - c);
end

% -------------------------------------------------------------------------
function y = onesLike(x)
% -------------------------------------------------------------------------
if isa(x, 'gpuArray')
    y = gpuArray.ones(size(x), classUnderlying(x));
else
    y = ones(size(x), 'like', x);
end
