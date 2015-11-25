function [y,mask] = vl_nndropout(x,varargin)
%VL_NNDROPOUT CNN dropout.
%   [Y,MASK] = VL_NNDROPOUT(X) applies dropout to the data X. MASK
%   is the randomly sampled dropout mask. Both Y and MASK have the
%   same size as X.
%
%   VL_NNDROPOUT(X, 'rate', R) sets the dropout rate to R.
%
%   [DZDX] = VL_NNDROPOUT(X, DZDY, 'mask', MASK) computes the
%   derivatives of the blocks projected onto DZDY. Note that MASK must
%   be specified in order to compute the derivative consistently with
%   the MASK randomly sampled in the forward pass. DZDX and DZDY have
%   the same dimesnions as X and Y respectivey.
%
%   Note that in the original paper on dropout, at test time the
%   network weights for the dropout layers are scaled down to
%   compensate for having all the neurons active. In this
%   implementation the dropout function itself already does this
%   compensation during training. So at test time no alterations are
%   required.

% Copyright (C) 2014-15 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

opts.rate = 0.5 ;
opts.mask = [] ;

backMode = numel(varargin) > 0 && ~isstr(varargin{1}) ;
if backMode
  dzdy = varargin{1} ;
  opts = vl_argparse(opts, varargin(2:end)) ;
else
  opts = vl_argparse(opts, varargin) ;
end

% determine mask
mask = opts.mask ;
scale = single(1 / (1 - opts.rate)) ;
if backMode && isempty(mask)
  warning('vl_nndropout: when using in backward mode, the mask should be specified') ;
end
if isempty(mask)
  if isa(x,'gpuArray')
    mask = scale * single(gpuArray.rand(size(x)) >= opts.rate) ;
  else
    mask = scale * single(rand(size(x)) >= opts.rate) ;
  end
end

% do job
if ~backMode
  y = mask .* x ;
else
  y = mask .* dzdy ;
end
