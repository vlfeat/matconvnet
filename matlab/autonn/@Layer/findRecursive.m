function [visited, selected, numSeen] = findRecursive(obj, what, n, depth, visited, selected, numSeen)
% FINDRECURSIVE Recursion on layers, used by FIND.
% WHAT, N, DEPTH: Search criteria (see FIND).
% VISITED: List of objects seen during recursion (which must be skipped).
% SELECTED: Boolean array (whether an object was selected or not).
% NUMSEEN: Number of valid entries in VISITED/SELECTED (for preallocation).

% Copyright (C) 2016 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).


  if n > 0 && nnz(selected) >= n
    return  % early break, already found the object we're after
  end

  % recurse on inputs that have not been visited yet (forward order)
  if depth > 0
    for i = 1:numel(obj.inputs)
      in = obj.inputs{i} ;
      if isa(in, 'Layer')
        valid = true ;
        for j = 1:numSeen
          if in == visited{j}  % already visited this object
            valid = false ;
            break ;
          end
        end
        if valid
          [visited, selected, numSeen] = obj.inputs{i}.findRecursive( ...
            what, n, depth - 1, visited, selected, numSeen) ;
        end
      end
    end
  end
  
  % add self to "visited" list
  numSeen = numSeen + 1 ;
  if numSeen > numel(visited)  % pre-allocate
    visited{end + 500} = [] ;
    selected(end + 500) = false ;
  end
  visited{numSeen} = obj ;

  % mark self as selected, if it matches the pattern
  if ischar(what)
    if any(what == '*') || any(what == '?')  % wildcards
      if ~isempty(regexp(obj.name, regexptranslate('wildcard', what), 'once'))
        selected(numSeen) = true ;
      end
    elseif isequal(obj.name, what) || isa(obj, what)
      selected(numSeen) = true ;
    end
  elseif isempty(what) || isequal(obj.func, what)
    selected(numSeen) = true ;
  end
end
