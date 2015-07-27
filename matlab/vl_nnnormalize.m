%VL_NNNORMALIZE CNN LRN normalization.
%   Y = VL_NNORMALIZE(X, PARAM) performs feature-wise sliding window
%   normalization of the image X. The normalized output is given by:
%
%     Y(i,j,k) = X(i,j,k) / L(i,j,k)^BETA
%
%   where the normalising factor is
%
%     L(i,j,k) = KAPPA + ALPHA * (sum_{q in Q(k)} X(i,j,k)^2,
%
%   PARAM = [N KAPPA ALPHA BETA], and N is the size of the window. The
%   window Q(k) itself is defined as:
%
%     Q(k) = [max(1, k-FLOOR((N-1)/2)), min(D, k+CEIL((N-1)/2))].
%
%   where D is the number of feature dimensions in X. Note in
%   particular that, by setting N >= 2D, the function can be used to
%   normalize the whole feature vector.
%
%   DZDX = VL_NNORMALIZE(X, PARAM, DZDY) computes the derivative of
%   the block projected onto DZDY. DZDX and DZDY have the same
%   dimensions as X and Y respectively.

% Copyright (C) 2014 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).
