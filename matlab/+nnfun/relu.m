function out = relu(x, dy, varargin)
%RELU ReLU
%   Y = RELU(X) applies ReLU to data X.
%
%   DX = RELU(X, DY) computes the derivative of the block projected onto
%   DY. DX and DY have the same dimensions as X and Y respectively.

opts.leak = 0;
opts = vl_argparse(opts, varargin, 'nonrecursive');

if opts.leak == 0
    if nargin < 2 || isempty(dy)
        % Forward
        out = max(x, 0);
    else
        % Backward
        out = dy.*(x > 0);
    end
else
    if nargin < 2 || isempty(dy)
        % Forward
        out = x.*(opts.leak + (1-opts.leak).*(x > 0));
    else
        % Backward
        out = dy.*(opts.leak + (1-opts.leak).*(x > 0));
    end
end
