function vl_setupnn()
%VL_SETUPNN Setup the MatConvNet toolbox.
%   VL_SETUPNN() function adds the MatConvNet toolbox to MATLAB path.

% Copyright (C) 2014-15 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

warning(sprintf(['AutoNN development has been moved to:\n\n' ...
  'https://github.com/vlfeat/autonn\n\n' ...
  '*** The "matconvnet/autodiff" branch will no longer be maintained. ***\n']));  %#ok<SPWRN>

root = vl_rootnn() ;
addpath(fullfile(root, 'matlab')) ;
addpath(fullfile(root, 'matlab', 'mex')) ;
addpath(fullfile(root, 'matlab', 'simplenn')) ;
addpath(fullfile(root, 'matlab', 'autonn')) ;
addpath(fullfile(root, 'matlab', 'xtest')) ;
addpath(fullfile(root, 'examples')) ;

if ~exist('gather')
  warning('The MATLAB Parallel Toolbox does not seem to be installed. Activating compatibility functions.') ;
  addpath(fullfile(root, 'matlab', 'compatibility', 'parallel')) ;
end

if numel(dir(fullfile(root, 'matlab', 'mex', 'vl_nnconv.mex*'))) == 0
  warning('MatConvNet is not compiled. Consider running `vl_compilenn`.');
end
