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
    weightDecay
    learningRate
    trainMethod
    value
  end
  properties (Constant)
    trainMethods = {'gradient', 'average'}  % list of methods, see CNN_TRAIN_AUTONN
  end
  
  methods
    function obj = Param(varargin)
      opts.weightDecay = 1 ;
      opts.learningRate = 1 ;
      opts.trainMethod = 'gradient' ;
      opts.value = [] ;
      opts.name = [] ;
      
      opts = vl_argparse(opts, varargin) ;
      
      obj.weightDecay = opts.weightDecay ;
      obj.learningRate = opts.learningRate ;
      obj.trainMethod = opts.trainMethod ;
      obj.value = opts.value ;
      obj.name = opts.name ;
    end
  end
end
