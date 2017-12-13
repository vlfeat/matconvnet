function out = softmaxy(z, dy, varargin)
%SOFTMAXY Softmax, derivative is a function of output y
%   Sum is performed along **1st** dimension.
%   Y = SOFTMAXY(X) applies softmax to data X.
%
%   DX = SOFTMAXY(X, DY) computes the derivative of the block projected onto
%   DY. DX and DY have the same dimensions as X and Y respectively.

if nargin < 2 || isempty(dy)
    % Forward
    xMax = max(z);
    z = z-xMax;
    ex = exp(z);
    out = ex./sum(ex, 1);
else
    % Backward
    out = z.*bsxfun(@minus, dy, sum(z.*dy, 1));
end
