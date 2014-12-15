%VL_IMREADJPEG Asynchronous multi-threaded jpeg file loading
%   VL_IMREADJPEG reads jpeg files with multiple threads and allows to
%   perform this operation asynchronously. In synchronous mode, 
%   ('Prefetch', false) this function creates pool of workers where each 
%   processes images from the queue and waits until all images are loaded.
%   In asynchronous mode ('Prefetch', true); images are added to the queue
%   but the function is not blocking and does not return any value.
%   Afterwards, images can be loaded with subsequent call without prefetch.
%
%   Synchrnous read:
%   IMGS = VL_IMREADJPEG(IMG_PATHS, 'NumThreads', NUM_THREADS) Read jpeg 
%   images from IMG_PATHS with NUM_THREADS separate threads. Each thread 
%   creates the image buffer and returns it in IMGS cell array. This
%   performs synchronous read (exit when all images read).
%
%   Asynchronous read:
%   VL_IMREADJPEG(IMG_PATHS, 'NumThreads', NUM_THREADS_P, 'Prefetch' true)
%   prefetch the jpeg images with NUM_THREADS worker threads. This command
%   exist directly when the jobs are queued. Prefetched images can be
%   loaded to Matlab by subsequent call of VL_IMREADJPEG(IMG_PATHS, 
%   'NumThreads', NUM_THREADS_L) which loads the prefetched images and waits
%   until the rest of the images are loaded. The number of prefetch threads
%   NUM_THREADS_P must be equal to number of loader threads NUM_THREADS_L
%   in order to keep the existing worker thread pool (with a different
%   values the workers pool is recreated which involves waiting until all
%   threads finish, this can lead to decreased performance).
%
% Copyright (C) 2014 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).