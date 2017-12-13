function out = reluy(z, dy, varargin)
%RELUY ReLU, derivative is a function of output y
%   Y = RELUY(X) applies ReLU to data X.
%
%   DX = RELUY(Y, DY) computes the derivative of the block projected onto
%   DY. DX and DY have the same dimensions as X and Y respectively.

opts.leak = 0;
opts = vl_argparse(opts, varargin, 'nonrecursive');

if opts.leak == 0
    if nargin < 2 || isempty(dy)
        % Forward
        out = max(z, 0);
    else
        % Backward
        out = dy.*sign(z);
    end
else
    if nargin < 2 || isempty(dy)
        % Forward
        out = z.*(opts.leak + (1-opts.leak).*(z > 0));
    else
        % Backward
        out = dy.*(opts.leak + (1-opts.leak).*(z > 0));
    end
end
