function y = vl_nnclip(x,xmin,xmax,dzdy)
%VL_NNCLIP Clipper.
%   Y = VL_NNCLIP(X,XMIN,MAX) clips X between XMIN and XMAX. X can have
%   arbitrary size. XMIN and XMAX can be empty, scalar, or matrix with same
%   size as X. Y have the same dimensions as X.
%
%   DZDX = VL_NNRELU(X,XMIN,XMAX, DZDY) computes the derivative of the block
%   projected onto DZDY. DZDX and DZDY have the same dimensions as
%   X and Y respectively.
%

if isempty(xmin)
    xmin=-inf;
end
if isempty(xmax)
    xmax=inf;
end
assert(isscalar(xmin)||isequal(size(x),size(xmin)),'xmin must be scalar or have same size as x');
assert(isscalar(xmax)||isequal(size(x),size(xmax)),'xmax must be scalar or have same size as x');

if nargin <= 3 || isempty(dzdy)
    y = min(max(x, xmin),xmax) ;
else
    y = dzdy .* (x > xmin & x < xmax) ;
end