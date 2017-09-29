function [h, t2i] = vl_tshow(T, varargin)
%VL_TSHOW Visualize a 4D tensor.
%   VL_TSHOW(T) shows the 4D tensor T in the current figure.
%
%   The tensor is shown as a montage of 2D slices (e.g. filters), with the
%   3rd dimension stacked along the rows and the 4th dimension along the
%   columns.
%
%   For effectively 3D tensors (i.e. 3rd or 4th dimension has size 1), the
%   slices are arranged on a 2D grid, to try to fill the axes as much as
%   possible. In this case the slices are column-major (i.e., start reading
%   on the top-left and go down).
%
%   VL_TSHOW(T, 'option', value, ...) accepts the following options:
%
%   `labels`:: true
%     If true, labels the x/y axis of the montage.
%
%   `aspectRatio`:: 3/4
%     Target aspect ratio when rearranging slices (default for 3D tensors).
%
%   `optimizeAR`:: []
%     Whether to rearrange slices to optimize aspect ratio: either true
%     (always do), false (never do), or empty (only for 3D tensors, the
%     default).
%
%   Any additional options are passed to IMAGESC (e.g. to set the parent
%   axes, or other properties).
%
%   H = VL_TSHOW(...) returns the image object's handle.
%
%   [H, T2I] = VL_TSHOW(...) also returns a function that converts tensor
%   coordinates to image-space: [row, col] = t2i(i, j, k, l) returns the
%   location of tensor element T(i,k,k,l) in the image. This can be useful
%   for additional plotting.

% Copyright (C) 2017 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

opts.labels = true ;
opts.optimizeAR = [] ;
opts.aspectRatio = 3/4 ;
[opts, varargin] = vl_argparse(opts, varargin, 'nonrecursive') ;

assert((isnumeric(T) || islogical(T)) && ndims(T) <= 4, ...
  'T must be a 1D to 4D numeric or logical tensor.') ;

assert(isempty(opts.optimizeAR) || (isscalar(opts.optimizeAR) && ...
  (isnumeric(opts.optimizeAR) || islogical(opts.optimizeAR))), ...
    'optimizeAR option must be either true, false or [].') ;

% Tensor size
T = gather(T) ;
origSz = size(T) ;
origSz(end+1:4) = 1 ;

if isequal(opts.optimizeAR, true) || ...
 (isempty(opts.optimizeAR) && (size(T,3) == 1 || size(T,4) == 1))
  % Redistribute 3rd/4th dim. to optimize aspect ratio if needed
  sz = optimizeAspectRatio(origSz, opts.aspectRatio) ;
  T = reshape(T, sz) ;
  if isempty(opts.optimizeAR)
    opts.labels = false ;
  end
else
  sz = origSz ;
end

% Stack input channels along rows (merge 1st dim. with 3rd), and output
% channels along columns (merge 2nd dim. with 4th), to form a 2D image
T = reshape(permute(T, [1 3 2 4]), sz(1) * sz(3), sz(2) * sz(4)) ;

% Display it
h = imagesc(T, varargin{:}) ;

ax = h.Parent ;
axis(ax, 'image') ;

% Display grid between filters
ax.XGrid = 'on';
ax.YGrid = 'on';
ax.GridAlpha = 1;
ax.XColor = 15/16 * [1 1 1];
ax.YColor = 15/16 * [1 1 1];
ax.TickLength = [0 0];
ax.XTickLabel = {};
ax.YTickLabel = {};

ax.YTick = sz(1) + 0.5 : sz(1) : sz(1) * sz(3) - 0.5;
ax.XTick = sz(2) + 0.5 : sz(2) : sz(2) * sz(4) - 0.5;

if opts.labels
  xlabel(sprintf('Fourth dimension (size %i)', origSz(4)), 'Parent', ax) ;
  ylabel(sprintf('Third dimension (size %i)', origSz(3)), 'Parent', ax) ;
end

% Restore label colors (made light-gray by setting X/YColor)
ax.XLabel.Color = 'k';
ax.YLabel.Color = 'k';

% If required, return handy function
if nargout > 1
  t2i = @(varargin) tensor2image(origSz, sz, varargin{:}) ;
end


function adjustedSz = optimizeAspectRatio(sz, aspect)

% Test aspect ratios of all possible combinations of 3rd/4th dim. stackings
n = prod(sz(3:4)) ;
cols = 1:n ;
rows = n ./ cols ;  % number of rows to tile N subplots

invalid = (rows ~= round(rows)) ;  % rule out those that do not divide evenly
rows(invalid) = [] ;
cols(invalid) = [] ;

asp = (rows * sz(1)) ./ (cols * sz(2)) ;  % subplots' aspect ratios
dist = abs(log(asp) - log(aspect)) ;  % logarithmic distance to preferred aspect ratio

% Choose best ratio
[~, i] = min(dist) ;
adjustedSz = [sz(1:2), rows(i), cols(i)] ;


function [row, col] = tensor2image(origSz, sz, i, j, k, l)
% Converts 4D tensor coordinates to image coordinates.

if nargin < 6, l = 1; end

if ~isequal(origSz, sz)  % rearranged to optimize aspect ratio
  % convert 3rd/4th dim. to a linear index, on the original tensor shape
  lin = k + (l - 1) * origSz(3) ;

  % now convert it to subscripts of the adjusted tensor shape
  k = mod(lin - 1, sz(3)) + 1 ;
  l = floor((lin - 1) / sz(3)) + 1 ;
end

% image-space coordinates corresponding to these tensor coordinates
row = i + (k-1) * sz(1);
col = j + (l-1) * sz(2);

