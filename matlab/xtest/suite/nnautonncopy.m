classdef nnautonncopy < nntest
  properties (TestParameter)
    topology = {'sequential', 'diamond'}
    sharedLayer = {0, 1, 2, 3}
  end
  properties
    layers
    answer
  end
  
  methods (TestClassSetup)
    function initNet(test)
      % sequential topology
      a = Input() ;
      b = sqrt(a) ;
      c = abs(b) ;
      d = c + 1 ;
      
      Layer.workspaceNames() ;
      test.layers.sequential = {a, b, c, d} ;
      test.answer.sequential = ...
        [1 1 1 1;  % which layers get copied with sharedLayer = 0
         0 1 1 1;  % same with sharedLayer = 1
         0 0 1 1;  % ...
         0 0 0 1;
         0 0 0 0] ;

      % diamond topology
      a = Input() ;
      b = sqrt(a) ;
      c = abs(a) ;
      d = b + c ;

      Layer.workspaceNames() ;
      test.layers.diamond = {a, b, c, d} ;
      test.answer.diamond = ...
        [1 1 1 1;  % different pattern due to diamond structure
         0 1 1 1;
         0 0 1 1;
         0 1 0 1;
         0 0 0 0];
    end
  end
  
  methods (Test)
    function testCopy(test, topology, sharedLayer)
      if strcmp(test.currentDataType, 'double') || strcmp(test.currentDevice, 'gpu'), return ; end
      
      % get network, sequence of layers, and correct answer
      sequence = test.layers.(topology) ;
      net = sequence{end} ;
      shouldBeCopied = test.answer.(topology) ;
      
      % call copy
      if sharedLayer == 0  % no shared layer
        other = net.deepCopy(@rename) ;
      else
        shared = sequence{sharedLayer} ;
        other = net.deepCopy(shared, @rename) ;
      end
      
      % get copied layers, in forward order
      otherSequence = other.find() ;
      
      assert(numel(otherSequence) == numel(sequence), ...
        'deepCopy or find failed to return the correct number of layers.') ;
      
      % check that only the correct layers were copied, and others remain
      % shared
      for i = 1:numel(sequence)
        assert(strcmp(sequence{i}.name, otherSequence{i}.name(1)), ...
          'deepCopy or find failed to return layers in the correct order') ;
        
        isCopied = strcmp(otherSequence{i}.name(2:end), '_copied') ;
        
        assert(isCopied == shouldBeCopied(sharedLayer + 1, i)) ;
      end
    end
  end
end

function name = rename(name)
  name = [name '_copied'] ;
end
