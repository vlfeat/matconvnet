classdef Param < Layer
  properties
    weightDecay
    learningRate
    value
  end
  
  methods
    function obj = Param(varargin)
      opts.weightDecay = 1 ;
      opts.learningRate = 1 ;
      opts.value = [] ;
      opts.name = [] ;
      
      opts = vl_argparse(opts, varargin) ;
      
      obj.weightDecay = opts.weightDecay ;
      obj.learningRate = opts.learningRate ;
      obj.value = opts.value ;
      obj.name = opts.name ;
    end
  end
end
