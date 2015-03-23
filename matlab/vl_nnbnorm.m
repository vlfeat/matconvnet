function [y,dzdg,dzdb] = vl_nnbnorm(x,g,b,varargin)
% VL_NNBNORM  CNN batch normalisation

% Copyright (C) 2015 Karel Lenc, Andrea Vedaldi
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

% ISSUE - needs to store internal state, another reason for having classes?

opts.epsilon = 1e-4;

backMode = numel(varargin) > 0 && ~ischar(varargin{1}) ;
if backMode
  dzdy = varargin{1} ;
  opts = vl_argparse(opts, varargin(2:end)) ;
else
  opts = vl_argparse(opts, varargin) ;
end

one = ones(1,1,'like',x);
eps = opts.epsilon;

x_sz = [size(x,1), size(x,2), size(x,3), size(x,4)];
% Create an array of size #channels x #samples
x = permute(x, [3 1 2 4]);
x = reshape(x, x_sz(3), []);

% do the job
u = mean(x, 2);
v = var(x, 0, 2);

v_nf = sqrt(v + one*eps); % variance normalisation factor
x_mu = bsxfun(@minus, x, u);
x_n = bsxfun(@times, x_mu, 1./v_nf);
%x_n = x;

if ~backMode
  y = bsxfun(@times, x_n, g);
  y = bsxfun(@plus, y, b);
else
  dzdy = permute(dzdy, [3 1 2 4]);
  dzdy = reshape(dzdy, x_sz(3), []);
  
  m = one * size(x, 2);
  dvdx = 2./(m - one) .* bsxfun(@minus, x_mu, one ./ m * sum(x_mu,2));
  
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