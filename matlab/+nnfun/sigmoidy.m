function out = sigmoidy(z, dy, varargin)
%SIGMOIDY Sigmoid, derivative is a function of output y
%   Y = SIGMOIDY(X) applies sigmoid to data X.
%
%   DX = SIGMOIDY(Y, DY) computes the derivative of the block projected onto
%   DY. DX and DY have the same dimensions as X and Y respectively.

if nargin < 2 || isempty(dy)
    % Forward
    y = 1./(1+exp(-z));
    out = y;
else
    % Backward
    out = dy.*z.*(1-z);
end
