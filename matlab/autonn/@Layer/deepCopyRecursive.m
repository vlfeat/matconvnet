function other = deepCopyRecursive(original, rename, visited)
% FINDRECURSIVE Recursion on layers, used by DEEPCOPY.

% Copyright (C) 2016 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  % create a shallow copy first
  other = original.copy() ;
  
  % rename if necessary
  other.name = rename(other.name) ;

  % pointer to the copied object, to be reused by any subsequent deep
  % copied layer that refers to the original object. this also marks it
  % as seen during the recursion.
  visited(original.id) = other ;


  % recurse on inputs
  for i = 1:numel(other.inputs)
    in = other.inputs{i} ;
    if isa(in, 'Layer')
      in.enableCycleChecks = false ;  % prevent cycle check when modifying a layer's input
      
      if visited.isKey(in.id)  % already seen/copied this original object
        other.inputs{i} = visited(in.id) ;  % use the copy
      else  % unseen/uncopied object, recurse on it and use the new copy
        other.inputs{i} = in.deepCopyRecursive(rename, visited) ;
      end
      
      in.enableCycleChecks = true ;
    end
  end
  
end
