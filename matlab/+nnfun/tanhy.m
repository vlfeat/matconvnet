function out = tanhy(z, dy, varargin)
%TANHY Tanh, derivative is a function of output y
%   Y = TANHY(X) applies sigmoid to data X.
%
%   DX = TANHY(Y, DY) computes the derivative of the block projected onto
%   DY. DX and DY have the same dimensions as X and Y respectively.

if nargin < 2 || isempty(dy)
    % Forward
    % ex = exp(-2.*z);
    % y = (1-ex)./(1+ex);
    out = tanh(z); % more stable
else
    % Backward
    out = dy.*(1-z.^2);
end
