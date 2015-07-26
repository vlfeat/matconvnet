classdef nnloss < nntest
  properties (TestParameter)
    loss = {'binarylog', 'hinge', 'logistic', 'log', 'softmaxlog', 'mhinge', 'mshinge'}
    weighed = {false, true}
  end

  properties
    x
  end

  methods
    function [x,c,instanceWeights] = getx(test,loss,weighed)
      numClasses = 3 ;
      numAttributes = 5 ;
      numImages = 3 ;
      w = 5 ;
      h = 4 ;
      switch loss
        case {'log', 'softmaxlog', 'mhinge', 'mshinge'}
          % multiclass
          instanceWeights = test.rand(h,w, 'single') / test.range / (h*w) ;
          c = single(randi(numClasses, h,w,1,numImages)) ;
        otherwise
          % binary
          instanceWeights = test.rand(h,w, numAttributes, 'single') / test.range / (h*w*numAttributes) ;
          c = single(sign(test.randn(h,w,numAttributes, numImages))) ;
      end
      switch loss
        case {'log'}
          x = test.rand(h,w, numClasses, numImages, 'single') / test.range * .80 + .10 ;
          x = bsxfun(@rdivide, x, sum(x,3)) ;
        case {'softmaxlog', 'mhinge', 'mshinge'}
          x = test.randn(h,w, numClasses, numImages, 'single') / test.range ;
        case {'hinge', 'logistic'}
          x = test.randn(h,w, numAttributes, numImages, 'single') / test.range ;
        case {'binarylog'}
          x = test.rand(h,w, numAttributes, numImages, 'single') / test.range * .80 + .10 ;
      end
    end
  end

  methods (Test)
    function convolutional(test, loss, weighed)
      [x,c,instanceWeights] = test.getx(loss) ;
      opts = {'loss',loss} ;
      if weighed, opts = {opts{:}, 'instanceWeights', instanceWeights} ; end
      y = vl_nnloss(x,c,[],opts{:}) ;
      dzdy = test.randn(size(y)) ;
      dzdx = vl_nnloss(x,c,dzdy,opts{:}) ;
      test.der(@(x) vl_nnloss(x,c,[],opts{:}), x, dzdy, dzdx, 5e-4, 2e-1) ;
    end
  end
end
