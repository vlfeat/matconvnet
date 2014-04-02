% GCONV
%    Y = GCONV(X, F)
%
%    X packs a number of images with value in a D-dimensional feature
%    space. X has dimension W x H x D x N, where (H,W) are the spatial
%    dimensions, D is the depth (number of feature channels) and
%    N the number of of images.
%
%    F packs a number of image filters. Y has dimension FW x FH x FD x K
%    where (FH,FW) are the filter spatial dimensions, FD is the depth
%    (number of feature channels) and K the number of filters.
%
%    F is compatible with X provided that 1 <= FH <= H, that 1 <= FW <= W,
%    and that D == FD.
%
%    Y packs the resulting filtered images in a YW x YH x K x N array.
%    Y has spatial dimensions (H-FH+1, W-FW+1),
%    depth K (equal to the number of filters), and packs N images.
%
%    Derivatives:
%
%    Y = GCONV(X,F)
%    y = S(Y)
%
%    d/dX S(GCONV(X,F)) = dS/dY * dGCONV/dX
%    d/dF S(GCONV(X,F)) = dS/dY * dGCONV/dF
