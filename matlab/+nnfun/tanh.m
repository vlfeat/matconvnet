function out = tanh(x, dy, varargin)
%TANH Tanh
%   Y = TANH(X) applies sigmoid to data X.
%
%   DX = TANH(X, DY) computes the derivative of the block projected onto
%   DY. DX and DY have the same dimensions as X and Y respectively.

% ex = exp(-2.*x);
% y = (1-ex)./(1+ex);
y = tanh(x); % more stable
if nargin < 2 || isempty(dy)
    % Forward
    out = y;
else
    % Backward
    out = dy.*(1-y.^2);
end
