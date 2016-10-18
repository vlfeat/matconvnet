function [other, visited, numVisited] = deepCopyRecursive(obj, shared, rename, visited, numVisited)
% FINDRECURSIVE Recursion on layers, used by DEEPCOPY.

% Copyright (C) 2016 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).


  % create a shallow copy first
  other = obj.copy() ;
  
  % rename if necessary
  other.name = rename(other.name) ;

  % pointer to the copied object, to be reused by any subsequent deep
  % copied layer that happens to share the same input
  obj.copied = other ;


  % recurse on inputs
  idx = other.getNextRecursion(visited, numVisited) ;
  for i = idx
    if ~any(cellfun(@(o) isequal(other.inputs{i}, o), shared))  % don't copy if shared
      
      other.inputs{i}.enableCycleChecks = false ;  % prevent cycle check when modifying a layer's input
      
      if ~isempty(other.inputs{i}.copied)  % reuse same deep copy
        other.inputs{i} = other.inputs{i}.copied ;
      else  % create a new one
        [other.inputs{i}, visited, numVisited] = ...
          other.inputs{i}.deepCopyRecursive(shared, rename, visited, numVisited) ;
      end
      
      other.inputs{i}.enableCycleChecks = true ;
    end
  end
  
  
  % repeat for test-mode inputs
  if ~isequal(other.testInputs, 'same')
    for i = 1:numel(other.testInputs)
      if isa(other.testInputs{i}, 'Layer') && ...
       ~any(cellfun(@(o) isequal(other.testInputs{i}, o), shared))

        if ~isempty(other.testInputs{i}.copied)  % reuse same deep copy
          other.testInputs{i} = other.testInputs{i}.copied ;
        else  % create a new one
          [other.testInputs{i}, visited, numVisited] = ...
            other.testInputs{i}.deepCopyRecursive(shared, rename, visited, numVisited) ;
        end
      end
    end
  end
  
  
  [visited, numVisited] = other.markRecursed(visited, numVisited) ;
end
