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
    function [x,c,dzdy,instanceWeights] = getx(test,loss)
      numClasses = 3 ;
      numAttributes = 5 ;
      numImages = 3 ;
      w = 5 ;
      h = 4 ;
      switch loss
        case {'log', 'softmaxlog', 'mhinge', 'mshinge', 'classerror'}
          % multiclass
          instanceWeights = test.rand(h,w, 'single') / test.range / (h*w) ;
          c = randi(numClasses, h,w,1,numImages) ;
          c = test.toDevice(c) ;
        otherwise
          % binary
          instanceWeights = test.rand(h,w, numAttributes, 'single') / test.range / (h*w*numAttributes) ;
          c = sign(test.randn(h,w,numAttributes, numImages)) ;
      end
      c = single(c) ;
      switch loss
        case {'log'}
          x = test.rand(h,w, numClasses, numImages, 'single') / test.range * .60 + .20 ;
          x = bsxfun(@rdivide, x, sum(x,3)) ;
        case {'binarylog'}
          x = test.rand(h,w, numAttributes, numImages, 'single') / test.range * .60 + .20 ;
        case {'softmaxlog'}
          x = test.randn(h,w, numClasses, numImages, 'single') / test.range ;
        case {'mhinge', 'mshinge', 'classerror'}
          x = test.randn(h,w, numClasses, numImages, 'single') / test.range ;
        case {'hinge', 'logistic', 'binaryerror'}
          x = test.randn(h,w, numAttributes, numImages, 'single') / test.range ;
      end
      dzdy = test.randn(1,1) / test.range ;
    end
  end

  methods (Test)
    function nullcategories(test, loss, weighed)
      [x,c,dzdy,instanceWeights] = test.getx(loss) ;
      % make a number of categories null
      c(:) = c(:) .* (test.randn(numel(c),1) > 0) ;
      opts = {'loss',loss} ;
      if weighed, opts = {opts{:}, 'instanceWeights', instanceWeights} ; end
      y = vl_nnloss(x,c,[],opts{:}) ;
      dzdx = vl_nnloss(x,c,dzdy,opts{:}) ;
      test.der(@(x) vl_nnloss(x,c,[],opts{:}), x, dzdy, dzdx, 0.001, -5e-1) ;
    end

    function convolutional(test, loss, weighed)
      [x,c,dzdy,instanceWeights] = test.getx(loss) ;
      opts = {'loss',loss} ;
      if weighed, opts = {opts{:}, 'instanceWeights', instanceWeights} ; end
      y = vl_nnloss(x,c,[],opts{:}) ;
      dzdx = vl_nnloss(x,c,dzdy,opts{:}) ;
      test.der(@(x) vl_nnloss(x,c,[],opts{:}), x, dzdy, dzdx, 0.001, -5e-1) ;
    end

  end
end
