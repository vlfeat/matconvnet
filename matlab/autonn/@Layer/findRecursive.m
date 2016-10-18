function [visited, selected, numVisited] = findRecursive(obj, what, n, depth, visited, selected, numVisited)
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
  
  if depth > 0
    % get indexes of inputs that have not been visited yet
    idx = obj.getNextRecursion(visited, numVisited) ;
    
    % recurse on them (forward order)
    for i = idx
      [visited, selected, numVisited] = obj.inputs{i}.findRecursive( ...
        what, n, depth - 1, visited, selected, numVisited) ;
    end
  end
  
  % mark this object as recursed (so it's not visited again)
  [visited, numVisited] = obj.markRecursed(visited, numVisited) ;
  
  if numVisited > numel(selected)  % pre-allocate selected list
    selected(end + 500) = false ;
  end

  % mark self as selected, if it matches the pattern
  if ischar(what)
    if any(what == '*') || any(what == '?')  % wildcards
      if ~isempty(regexp(obj.name, regexptranslate('wildcard', what), 'once'))
        selected(numVisited) = true ;
      end
    elseif isequal(obj.name, what) || isa(obj, what)
      selected(numVisited) = true ;
    end
  elseif isempty(what) || isequal(obj.func, what)
    selected(numVisited) = true ;
  end
end
