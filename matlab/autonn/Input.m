classdef Input < Layer
%Input
%   Defines a network input (such as images or labels).

% Copyright (C) 2016 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  properties
    gpu
  end
  
  methods
    function obj = Input(varargin)
      opts.name = [] ;
      opts.gpu = false ;
      
      if isscalar(varargin) && ischar(varargin{1})
        opts.name = varargin{1} ;  % special syntax, just pass in the name (possibly deprecate in the future?)
      else
        opts = vl_argparse(opts, varargin, 'nonrecursive') ;
      end
      
      obj.name = opts.name ;
      obj.gpu = opts.gpu ;
    end
  end
end
