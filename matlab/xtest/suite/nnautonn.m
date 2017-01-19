classdef nnautonn < nntest
  properties (TestParameter)
    topology = {'sequential', 'diamond'}
  end

  methods (Test)
    function testEval(test, topology)
      if strcmp(test.currentDataType, 'double'), return ; end
      
      x = Input() ;
      
      switch topology
      case 'sequential'
        % a sequence of 3 layers that can be worked out by hand.
        
        % network input
        x_value = [1, -1] ;
        x_value = reshape(single(x_value), 1, 1, 2) ;
        
        % expected output
        y_value = [1, 2] ;
        y_value = reshape(single(y_value), 1, 1, 2) ;
        
        % expected derivative
        x_der = [-1, 0] ;
        x_der = reshape(single(x_der), 1, 1, 2) ;
        
        % derivative for backprop
        y_der = ones(1, 1, 2, 'single') ;
        
        % convs will just be the identity, or its negative
        w_value = reshape(eye(2, 'single'), 1, 1, 2, 2) ;
        
        y = vl_nnconv(x, Param('value', w_value), []) ;
        y = vl_nnrelu(y) ;
        y = vl_nnconv(y, Param('value', -w_value), Param('value', single([2, 2]))) ;

      case 'diamond'
        % input branches out into 2 middle layers, which are then joined
        % again. tests correct accumulation of derivatives, and math
        % operators.
        
        % network input
        x_value = [3; 10] ;
        
        % expected output
        y_value = [9; 30] ;
        
        % expected derivative
        x_der = [3; 3] ;
        
        % derivative for backprop
        y_der = ones(2, 1) ;
        
        % ensure they're matrix multiplies, instead of scalar
        x1 = 2 * eye(2) * x ;
        x2 = 5 * eye(2) * x ;
        y = x2 - x1 ;
        
      otherwise
        error('Unknown topology.')
      end
      
      % name layers, create net and set input
      Layer.workspaceNames() ;
      net = Net(y) ;
      net.setInputs('x', x_value) ;
      
      % handle GPU
      if strcmp(test.currentDevice, 'gpu')
        gpuDevice(1) ;
        net.move('gpu') ;
        y_der = gpuArray(y_der) ;
      end
      
      % run forward and backward
      net.eval('normal', y_der) ;
      
      % check output and input derivatives
      disp('Output:') ;
      disp(squeeze(net.getValue(y))) ;
      disp('Input derivative:') ;
      disp(squeeze(net.getDer(x))) ;
      
      test.eq(net.getValue(y), y_value) ;
      test.eq(net.getDer(x), x_der) ;
    end
  end
end
