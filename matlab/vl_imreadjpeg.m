%VL_IMREADJPEG (A)synchronous multithreaded JPEG image loader.
%   IMAGES = VL_IMREADJPEG(FILES) reads the specified cell array
%   FILES of JPEG files into the cell array of images IMAGES.
%
%   IMAGES = VL_IMREADJPEG(FILES, 'NumThreads', T) uses T parallel
%   threads to accelerate the operation. Note that this is
%   independent of the number of computational threads used by
%   MATLAB.
%
%   VL_IMREADJPEG(FILES, 'Prefetch') starts reading the specified
%   images but returns immediately to MATLAB. Reading happens
%   concurrently with MATLAB in one or more separated threads.  A
%   subsequent call IMAGES=VL_IMREADJPEG(FILES) *specifying exactly
%   the same files in the same order* will then return the loaded
%   images. This can be sued to quickly load a batch of JPEG images
%   as MATLAB is busy doing something else.
%
%   The function takes the following options:
%
%   Prefetch:: [not specified]
%     If specified, run without blocking (see above).
%
%   Verbose:: [not specified]
%     Increase the verbosity level.
%
%   NumThreads:: [1]
%     Specify the number of threads used to read images. This number
%     must be at least 1. Note that it does not make sense to specify
%     a number larger than the number of available CPU cores, and
%     often fewer threads are sufficient as reading images is memory
%     access bound rather than CPU bound.

% Copyright (C) 2014-15 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).