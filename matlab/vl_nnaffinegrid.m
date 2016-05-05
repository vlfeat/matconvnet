function y = vl_nnaffinegrid(x, sz, dzdy)
%VL_NNAFFINEGRID Affine grid generator.
%   Y = VL_NNAFFINEGRID(X, SZ) generates an affine grid for a spatial
%   transformer (see VL_NNBILINEARSAMPLER). SZ is the size of the images
%   (2 elements vector of height and width).
%
%   Alternatively, SZ can be a pre-computed H*W x 2 grid, in the format
%   accepted by VL_NNBILINEARSAMPLER.
%
%   X is a 1x1x6xN TENSOR corresponding to:
%    [ c1 c2 c5 ]
%    [ c3 c4 c6 ]
%    [  0  0  1 ]
%   i.e., [x_out] = [c1 c2]  * [x_in] + [c5]
%         [y_out]   [c3 c4]    [y_in]   [c6]
%   Y is a HoxWox2xN grid which corresponds to applying the above affine
%   transform to the [-1,1] normalized x,y coordinates.
%
%   DZDX = VL_NNAFFINEGRID(X, SZ, DZDY) computes the derivative of the
%   block projected onto DZDY. DZDX and DZDY have the same dimensions as
%   X and Y respectively.

% Copyright (C) 2016 Ankush Gupta, Joao Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).


  useGPU = isa(x, 'gpuArray');

  % generate the grid coordinates:
  if numel(sz) == 2
    Ho = sz(1);
    Wo = sz(2);
    xi = linspace(-1, 1, Ho);
    yi = linspace(-1, 1, Wo);

    [yy,xx] = meshgrid(xi,yi);
    xxyy = [yy(:), xx(:)] ; % Mx2
    if useGPU
      xxyy = gpuArray(xxyy);
    end
  else
    % reuse a cached grid
    assert(ndim(sz) == 3 && size(sz,3) == 2, ['SZ must either be a ' ...
      'vector [H,W] with the image size, or a H x W x 2 grid.']);
    Ho = size(sz,1);
    Wo = size(sz,2);
    xxyy = reshape(sz, [], 2);
    if useGPU
      assert(isa(xxyy, 'gpuArray'), 'Grid is not a gpuArray.');
    end
  end


  if nargin < 3
    % forward pass
    % reshape the tfm params into matrices:
    nbatch = size(x,4);
    A = reshape(x, 2,3,nbatch);
    L = A(:,1:2,:);
    L = reshape(L,2,2*nbatch); % linear part

    % transform the grid:
    t = A(:,3,:); % translation
    t = reshape(t,1,2*nbatch);
    g = bsxfun(@plus, xxyy * L, t); % apply the transform
    g = reshape(g, Wo, Ho, 2, nbatch);

    % cudnn compatibility:
    y = permute(g, [3,2,1,4]);
    
  else
    % backward pass
    nbatch = size(dzdy,4);

    % cudnn compatibility:
    dY = permute(dzdy, [3,2,1,4]);

    % create the gradient buffer:
    dA = zeros([2,3,nbatch], 'single');
    if useGPU, dA = gpuArray(dA); end

    dY = reshape(dY, Ho*Wo, 2*nbatch);
    % gradient wrt the linear part:
    dL = xxyy' * dY;
    dL = reshape(dL,2,2,nbatch);
    dA(:,1:2,:) = dL;

    % gradient wrt translation (or bias):
    dt = reshape(sum(dY,1),2,1,nbatch);
    dA(:,3,:) = dt;

    y = reshape(dA, size(x));
  end
end

