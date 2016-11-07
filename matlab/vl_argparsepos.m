function [opts, args] = vl_argparsepos(opts, args, varargin)
%VL_ARGPARSEPOS
%   Same as VL_ARGPARSE, but allows arbitrary positional arguments before
%   the name-value pairs.
%
%   Example:
%     opts.pad = 0 ;
%     [opts, args] = vl_argparsepos(opts, {x, 'pad', 1, 'unknown', []}) ;
%   The result will be:
%     opts.pad = 1
%     args = {x, 'unknown', []}

% Copyright (C) 2016 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  % even or odd indexes, always including the 2nd-to-last element of args
  idx = (numel(args) - 1 : -2 : 1) ;  % reverse order
  
  if isempty(idx)
    % no name-value pairs in the list
    firstPair = numel(args) + 1 ;
  else
    % find first invalid name-value pair, starting from the end
    pos = find(~cellfun(@ischar, args(idx)), 1) ;

    if isempty(pos)  % all are valid
      firstPair = idx(end) ;
    else  % map back to argument indexes
      firstPair = idx(pos) + 2 ;
    end
  end
  
  % separate them
  namedArgs = args(firstPair:end) ;
  posArgs = args(1:firstPair-1) ;

  % call vl_argparse
  if nargout == 2
    [opts, namedArgs] = vl_argparse(opts, namedArgs, varargin{:}) ;
    args = [posArgs, namedArgs] ;  % back together
  else
    opts = vl_argparse(opts, namedArgs, varargin{:}) ;
  end
  
end

