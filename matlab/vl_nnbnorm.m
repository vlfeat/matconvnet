%VL_NNBNORM CNN batch normalisation.
%   Y = VL_NNBNORM(X,G,B) computes the batch normalization of the
%   input X. This is defined as:
%
%      Y(i,j,k,t) = G(k) * (X(i,j,k,t) - mu(k)) / sigma(k) + B(k)
%
%  where:
%
%      mu(k) = mean_ijt X(i,j,k,t),
%      sigma2(k) = mean_ijt (X(i,j,k,t) - mu(k))^2,
%      sigma(k) = sqrt(sigma2(k) + EPSILON)
%
%   are respectively the per-channel mean, variance, and standard
%   deviation of a channel of the data X and G(k) and B(k) define
%   respectively a multiplicative and additive constant to scale each
%   data channel. Note that statistics are accumulated across all the
%   items of the batch in the 4D tensor X (from which the name batch
%   normalization). The constant EPSILON is used to regularize the
%   computation of sigma(k) and avoid divisions by zero or very small
%   numbers.
%
%   [DZDX,DZDG,DZDB] = VL_NNBNORM(X,G,B,DZDY) computes the derviatives
%   of the block projected onto DZDY. DZDX, DZDG, DZDB and DZDY have
%   the same dimensions as X, G, B, and Y respectivey.
%
%   Optionally, [Y,MEAN,MOMENTS] = VL_NNBNORM(...) and
%   [DZDX,DZDG,DZDB,MOMENTS] = VL_NNBNORM(...,DZDY) return the values
%   of the vectors mu and sigma in the formulas above. Here, MOMENTS
%   is a 2 x DEPTH array, where the vectors mu and sigma are the rows.
%
%   VL_NNBNROM(..., 'Option', value) takes the following options:
%
%   `Epsilon`:: 1e-4
%       Specifies the constant EPSILON in the formuals above.
%
%   `Moments`:: unspecified
%       Specifies an array MOMENTS with the values of mu and sigma to
%       use instead of computing them according to the equations
%       above. This is useful to disable batch normalization during
%       testing.
%
%   See also: VL_NNNORMALIZE().

% Copyright (C) 2015 Sébastien Ehrhardt, Karel Lenc and Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).
