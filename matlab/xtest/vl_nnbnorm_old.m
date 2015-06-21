function [y,dzdg,dzdb] = vl_nnbnorm(x,g,b,varargin)
% VL_NNBNORM  CNN batch normalisation
%    Y = VL_NNBNORM(X,G,B) computes the batch normalization of the
%    input X. This is defined as
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
%   Epsilon:: 1e-4
%       Specify the EPSILON constant.
%
%   See also: VL_NNNORMALIZE().

% Copyright (C) 2015 Karel Lenc.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

% ISSUE - needs to store internal state, another reason for having classes?

% -------------------------------------------------------------------------
%                                                             Parse options
% -------------------------------------------------------------------------

opts.epsilon = 1e-4 ;
backMode = numel(varargin) > 0 && ~ischar(varargin{1}) ;
if backMode
  dzdy = varargin{1} ;
  opts = vl_argparse(opts, varargin(2:end)) ;
else
  opts = vl_argparse(opts, varargin) ;
end

% -------------------------------------------------------------------------
%                                                                    Do job
% -------------------------------------------------------------------------

one = ones(1,1,'like',x);
eps = opts.epsilon;

x_sz = [size(x,1), size(x,2), size(x,3), size(x,4)];
% Create an array of size #channels x #samples
x = permute(x, [3 1 2 4]);
x = reshape(x, x_sz(3), []);

% do the job
u = mean(x, 2);
v = var(x, 1, 2);

v_nf = sqrt(v + opts.epsilon) ;
x_mu = bsxfun(@minus, x, u);
x_n = bsxfun(@times, x_mu, 1./v_nf);

if ~backMode
  y = bsxfun(@times, x_n, g);
  y = bsxfun(@plus, y, b);
else
  dzdy = permute(dzdy, [3 1 2 4]);
  dzdy = reshape(dzdy, x_sz(3), []);

  m = one * size(x, 2);
  dvdx = 2./(m - 0*one) .* bsxfun(@minus, x_mu, one ./ m * sum(x_mu,2));

  v_nf_d = -0.5 * (v + one*eps) .^ (-3/2);

  dzdx = bsxfun(@times, bsxfun(@minus, dzdy, one ./ m * sum(dzdy,2)), 1./v_nf(:));
  dzdx = dzdx + bsxfun(@times, bsxfun(@times, dvdx, sum(dzdy .* x_mu, 2)), v_nf_d);
  %dzdx = dzdy;
  y = bsxfun(@times, dzdx, g);
  dzdg = sum(dzdy .* x_n, 2);
  dzdb = sum(dzdy, 2);
end

y = reshape(y, x_sz(3), x_sz(1), x_sz(2), x_sz(4));
y = permute(y, [2 3 1 4]);

end