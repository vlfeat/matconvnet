classdef nnautonnfind < nntest
  properties (TestParameter)
    topology = {'sequential', 'diamond'}
    index = {1, 2, 3, 4, 'all', -1}
    criteria = {'index', 'name', 'function'}
  end
  properties
    layers
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

      % diamond topology
      a = Input() ;
      b = sqrt(a) ;
      c = abs(a) ;
      d = b + c ;

      Layer.workspaceNames() ;
      test.layers.diamond = {a, b, c, d} ;
    end
  end
  
  methods (Test)
    function testFind(test, topology, criteria, index)
      if strcmp(test.currentDataType, 'double') || strcmp(test.currentDevice, 'gpu'), return ; end
      
      % get network and correct sequence of layers
      sequence = test.layers.(topology) ;
      net = sequence{end} ;
      
      if isnumeric(index)
        % get one layer, and check it's the right one.
        if index >= 1  % the correct layer, according to the sequence
          correct = sequence{index} ;
        elseif index == -1  % searching backwards
          correct = sequence{end} ;
        end
        
        % call find
        switch criteria
        case 'index'
          found = net.find(index) ;
          
        case 'name'
          name = correct.name ;
          found = net.find(name, 1) ;
          
        case 'function'
          f = correct.func ;
          if isempty(f), f = 'Input'; end
          found = net.find(f, 1) ;
        end
        
        disp(['Found: ' found.name]) ;
        assert(strcmp(found.name, correct.name)) ;
        
      elseif isequal(index, 'all')
        % find all layers, and check if they're in the correct order
        found = net.find() ;
        
        assert(numel(found) == numel(sequence)) ;
        for i = 1:numel(found)
          assert(strcmp(found{i}.name, sequence{i}.name)) ;
        end
      end
    end
  end
end
