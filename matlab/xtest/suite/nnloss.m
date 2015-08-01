classdef nnloss < nntest
  properties (TestParameter)   
    loss = {...
      'classerror', 'log', 'softmaxlog', 'mhinge', 'mshinge', ...
      'binaryerror', 'binarylog', 'logistic', 'hinge'}
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
        case {'log', 'softmaxlog', 'mhinge', 'mshinge', 'classerror'}
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
        case {'binarylog'}
          x = test.rand(h,w, numAttributes, numImages, 'single') / test.range * .80 + .10 ;
        case {'softmaxlog', 'mhinge', 'mshinge', 'classerror'}
          x = test.randn(h,w, numClasses, numImages, 'single') / test.range ;
        case {'hinge', 'logistic', 'binaryerror'}
          x = test.randn(h,w, numAttributes, numImages, 'single') / test.range ;     
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
