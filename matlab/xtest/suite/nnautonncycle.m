classdef nnautonncycle < nntest
  methods (Test)
    function testCycle(test)
      % diamond topology
      a = Input() ;
      b = sqrt(a) ;
      c = abs(a) ;
      d = b + c ;
      e = exp(d) ;
      
      % introduce cycles in the DAG
      try
        b.inputs{1} = e ;
      catch err
        assert(strcmp(err.identifier, 'MatConvNet:CycleCheckFailed'), ...
          'Failed to catch cycle in DAG.') ;
      end
      
      try
        d.inputs{2} = d ;
      catch err
        assert(strcmp(err.identifier, 'MatConvNet:CycleCheckFailed'), ...
          'Failed to catch cycle in DAG.') ;
      end
    end
  end
end
