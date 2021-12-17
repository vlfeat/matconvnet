function vl_setupnn()
%VL_SETUPNN Setup the MatConvNet toolbox.
%   VL_SETUPNN() function adds the MatConvNet toolbox to MATLAB path.

% Copyright (C) 2014-15 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

root = fileparts(which('vl_setupnn'));
addpath(fullfile(root)) ;
addpath(fullfile(root, 'mex')) ;
addpath(fullfile(root, 'simplenn')) ;
addpath(fullfile(root, 'xtest')) ;
addpath(fullfile(root, '..', 'examples')) ;

if ~exist('gather.m', 'file')
  warning('The MATLAB Parallel Toolbox does not seem to be installed. Activating compatibility functions.') ;
  addpath(fullfile(root, 'matlab', 'compatibility', 'parallel')) ;
end

if ~exist(['vl_nnconv.', mexext()], 'file')
  warning('MatConvNet is not compiled. Consider running `vl_compilenn`.');
end
