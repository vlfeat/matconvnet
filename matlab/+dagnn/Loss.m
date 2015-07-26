classdef Loss < dagnn.ElementWise
  properties
    loss = 'softmaxlog'
  end

  properties (Transient)
    average = 0
    numAveraged = 0
  end

  methods
    function outputs = forward(obj, inputs, params)
      outputs{1} = vl_nnloss(inputs{1}, inputs{2}, [], 'loss', obj.loss) ;
      n = obj.numAveraged ;
      m = n + size(inputs{1},4) ;
      obj.average = (n * obj.average + gather(outputs{1})) / m ;
      obj.numAveraged = m ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs{1} = vl_nnloss(inputs{1}, inputs{2}, derOutputs{1}, 'loss', obj.loss) ;
      derInputs{2} = [] ;
      derParams = {} ;
    end

    function obj = Loss(varargin)
      obj.load(varargin) ;
    end

    function reset(obj)
      obj.average = 0 ;
      obj.numAveraged = 0 ;
    end
  end
end
