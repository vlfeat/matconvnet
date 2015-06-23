% VL_NNBNORM  CNN batch normalisation
%   Y = VL_NNBNORM(X,G,B) computes the batch normalization of the
%   input X. This is defined as:
%
%      Y(i,j,k,t) = G(k) * (X(i,j,k,t) - mu(k)) / sigma(k) + B(k)
%
%   where
%
%      mu(k) = mean_ijt X(i,j,k,t),
%      sigma(k) = sqrt(sigma2(k) + EPSILON),
%      sigma2(k) = mean_ijt (X(i,j,k,t) - mu(k))^2
%
%   are respectively the per-channel mean, standard deviation, and
%   variance of the input and G(k) and B(k) define respectively a
%   multiplicative and additive constant to scale each input
%   channel. Note that statistics are computed across all feature maps
%   in the batch packed in the 4D tensor X. Note also that the
%   constant EPSILON is used to regularize the computation of sigma(k)
%
%   [Y,DZDG,DZDB] = VL_NNBNORM(X,G,B,DZDY) computes the derviatives of
%   the output Z of the network given the derivatives with respect to
%   the output Y of this function.
%
%   VL_NNBNROM(..., 'Option', value) takes the following options:
%
%   `Epsilon`:: 1e-4
%       Specify the EPSILON constant.
%
%   See also: VL_NNNORMALIZE().

% Copyright (C) 2015 Karel Lenc and Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).
