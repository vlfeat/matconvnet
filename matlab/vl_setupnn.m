function vl_setupnn()
% VL_SETUPNN  Setup VLNN toolbox
%   The function adds the VLNN toolbox to MATLAB path.

% Author: Andrea Vedaldi

root = vl_rootnn() ;
addpath(fullfile(root, 'matlab')) ;
addpath(fullfile(root, 'matlab', 'mex')) ;
addpath(fullfile(root, 'matlab', 'xtest')) ;
