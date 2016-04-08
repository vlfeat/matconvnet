classdef Param < Layer
  properties
    weightDecay
    learningRate
    trainMethod
    value
  end
  properties (Constant)
    trainMethods = {'gradient', 'average'}
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
