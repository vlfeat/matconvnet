classdef Scale < dagnn.ElementWise
  properties
    size
    hasBias = true
  end

  methods

    function outputs = forward(obj, inputs, params)
      args = horzcat(inputs, params) ;
      outputs{1} = bsxfun(@times, args{1}, args{2}) ;
      if obj.hasBias
        outputs{1} = bsxfun(@plus, outputs{1}, args{3}) ;
      end
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      args = horzcat(inputs, params) ;
      sz = [size(args{2}) 1 1 1 1] ;
      sz = sz(1:4) ;
      dargs{1} = bsxfun(@times, derOutputs{1}, args{2}) ;
      dargs{2} = derOutputs{1} .* args{1} ;
      for k = find(sz == 1)
        dargs{2} = sum(dargs{2}, k) ;
      end
      if obj.hasBias
        dargs{3} = derOutputs{1} ;
        for k = find(sz == 1)
          dargs{3} = sum(dargs{3}, k) ;
        end
      end
      derInputs = dargs(1:numel(inputs)) ;
      derParams = dargs(numel(inputs)+(1:numel(params))) ;
    end

    function obj = Scale(varargin)
      obj.load(varargin) ;
    end
    
    function set.size(obj, ksize)
      % make sure that ksize has 4 dimensions
      ksize = [ksize(:)' 1 1 1 1] ;
      obj.size = ksize(1:4) ;
    end
    
    function params = initParams(obj)
      sc = sqrt(2 / prod(obj.size(1:3))) ;
      params{1} = randn(obj.size,'single') * sc ;
      if obj.hasBias
        params{2} = zeros(obj.size(4),1,'single') * sc ;
      end
    end
  end
end
