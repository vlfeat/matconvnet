function out = softmax(x, dy, varargin)
%SOFTMAX Softmax
%   Sum is performed along **1st** dimension.
%   Y = SOFTMAX(X) applies softmax to data X.
%
%   DX = SOFTMAX(X, DY) computes the derivative of the block projected onto
%   DY. DX and DY have the same dimensions as X and Y respectively.

xMax = max(x);
x = x-xMax;
ex = exp(x);
y = ex./sum(ex, 1);
if nargin < 2 || isempty(dy)
    % Forward
    out = y;
else
    % Backward
    out = y.*bsxfun(@minus, dy, sum(y.*dy, 1));
end
