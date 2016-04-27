classdef nnautonnlayers < nntest
  methods (Test)
    function testLayers(test)
      % use Params for all inputs so we can choose their values now
      x = Param('value', randn(7, 7, 2, 5, test.currentDataType)) ;
      w = Param('value', randn(3, 3, 2, 2, test.currentDataType)) ;
      b = Param('value', randn(2, 1, test.currentDataType)) ;
      labels = Param('value', ones(5, 1)) ;
      Layer.workspaceNames() ;
      
      % test several layers and syntaxes
      
      do(test, vl_nnrelu(x)) ;
      
      do(test, vl_nnconv(x, w, b)) ;
      
      do(test, vl_nnconv(x, w, b, 'stride', 3, 'pad', 2)) ;
      
      do(test, vl_nnpool(x, 2)) ;
      
      do(test, vl_nnpool(x, [2, 2], 'stride', 2, 'pad', 1)) ;
      
      do(test, vl_nndropout(x, 'rate', 0.1)) ;
      
      do(test, vl_nnloss(x, labels, 'loss', 'classerror')) ;
      
      if strcmp(test.currentDataType, 'single')
        % bnorm parameters are created as single
        do(test, vl_nnbnorm(x)) ;
      end
    end
    
    function testMath(test)
      % use Params for all inputs so we can choose their values now
      a = Param('value', randn(3, 3, test.currentDataType)) ;
      b = Param('value', randn(3, 3, test.currentDataType)) ;
      c = Param('value', randn(1, 1, test.currentDataType)) ;
      d = Param('value', randn(3, 1, test.currentDataType)) ;
      Layer.workspaceNames() ;
      
      % test several operations
      
      % wsum
      do(test, a + b) ;
      do(test, 10 * a) ;
      do(test, a + 2 * b - c) ;  % collected arguments in a single wsum
      
      % matrix
      do(test, a * b) ;
      do(test, a') ;
      
      % binary with expanded dimensions
      do(test, a .* d) ;
      do(test, a .^ 2) ;
    end
    
    function testConv(test)
      % extra conv tests
      if strcmp(test.currentDataType, 'double'), return, end
      
      x = Param('value', randn(7, 7, 2, 5, test.currentDataType)) ;
      
      % 'size' syntax
      do(test, vl_nnconv(x, 'size', [7, 7, 2, 5])) ;
      
      % bias
      layer = vl_nnconv(x, 'size', [7, 7, 2, 5], 'hasBias', false) ;
      do(test, layer) ;
      test.verifyEmpty(layer.inputs{3}) ;
      
      % Param learning arguments
      layer = vl_nnconv(x, 'size', [7, 7, 2, 5], ...
          'learningRate', [1, 2], 'weightDecay', [3, 4]) ;
      do(test, layer) ;
      test.eq(layer.inputs{2}.learningRate, 1) ;
      test.eq(layer.inputs{3}.learningRate, 2) ;
      test.eq(layer.inputs{2}.weightDecay, 3) ;
      test.eq(layer.inputs{3}.weightDecay, 4) ;
    end
  end
  
  methods
    function do(test, output)
      % show layer for debugging
      display(output) ;
      
      % compile net
      net = Net(output) ;
      
      % run forward only
      net.eval('forward') ;
      
      % check output is non-empty
      y = net.getValue(output) ;
      test.verifyNotEmpty(y) ;
      
      % create derivative with same size as output
      der = randn(size(net.getValue(output)), test.currentDataType) ;
      
      % handle GPU
      if strcmp(test.currentDevice, 'gpu')
        gpuDevice(1) ;
        net.move('gpu') ;
        der = gpuArray(der) ;
      end
      
      % run forward and backward
      net.eval('normal', der) ;
      
      % check all derivatives are non-empty
      ders = net.getValue(2:2:numel(net.vars)) ;
      for i = 1:numel(ders)
        test.verifyNotEmpty(ders{i}) ;
      end
    end
  end
end
