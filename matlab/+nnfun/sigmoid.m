function out = sigmoid(x, dy, varargin)
%SIGMOID Sigmoid
%   Y = SIGMOID(X) applies sigmoid to data X.
%
%   DX = SIGMOID(X, DY) computes the derivative of the block projected onto
%   DY. DX and DY have the same dimensions as X and Y respectively.

y = 1./(1+exp(-x));
if nargin < 2 || isempty(dy)
    % Forward
    out = y;
else
    % Backward
    out = dy.*y.*(1-y);
end
