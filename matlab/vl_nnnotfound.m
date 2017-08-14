function vl_nnnotfound(fname)
%VL_NNNOTFOUND Prints a help error message to set up MatConvNet
%  Warn users about common pitfalls in setting up MatConvNet.

% Copyright (C) 2017 Karel Lenc.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

paths = strsplit(path, pathsep) ;
mexpath = fullfile(fileparts(mfilename('fullpath')), 'mex') ;

if ~exist(mexpath, 'dir') || ~exist(fullfile(mexpath, [fname, '.', mexext]), 'file')
  error('MatConvNet not compiled or the compilation fialed. Please run `vl_compilenn`.');
end

if ~ismember(mexpath, paths)
  error('MatConvNet not set up. Please run \n\t`run %s; rehash;`.', ...
    fullfile(vl_rootnn(), 'matlab', 'vl_setupnn.m')) ;
end

if strcmp(pwd, fullfile(vl_rootnn, 'matlab'))
  error(['MatConvNet cannot be run in MatConvNet''s MATLAB path %s.\n', ...
    'Please change path and call `rehash`.%s'], vl_rootnn()) ;
end
