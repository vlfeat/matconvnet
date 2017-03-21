classdef Param < Layer
%Param
%   Defines a network parameter (such as a convolution's weights).
%   Note that some functions (e.g. vl_nnconv, vl_nnbnorm) can create and
%   initialize Param objects automatically. Such behavior is optional,
%   since they can use any other layer's output in their arguments, not
%   just Params.

% Copyright (C) 2016 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  properties
    value
    weightDecay
    learningRate
    trainMethod
    gpu
  end
  properties (Constant)
    trainMethods = {'gradient', 'average', 'none'}  % list of methods, see CNN_TRAIN_AUTONN
  end
  
  methods
    function obj = Param(varargin)
      opts.name = [] ;
      opts.value = [] ;
      opts.weightDecay = 1 ;
      opts.learningRate = 1 ;
      opts.trainMethod = 'gradient' ;
      opts.gpu = true ;
      
      opts = vl_argparse(opts, varargin, 'nonrecursive') ;
      
      obj.name = opts.name ;
      obj.value = opts.value ;
      obj.weightDecay = opts.weightDecay ;
      obj.learningRate = opts.learningRate ;
      obj.trainMethod = opts.trainMethod ;
      obj.gpu = opts.gpu ;
    end
  end
end
 