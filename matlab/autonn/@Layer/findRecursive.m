function selected = findRecursive(obj, what, n, depth, visited, selected)
% FINDRECURSIVE Recursion on layers, used by FIND.
% WHAT, N, DEPTH: Search criteria (see FIND).
% VISITED: Dictionary of objects seen during recursion so far.
% SELECTED: Cell array of selected objects.

% Copyright (C) 2016 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  
  if depth > 0
    % get indexes of inputs that have not been visited yet
    idx = obj.getNextRecursion(visited) ;
    
    % recurse on them (forward order)
    for i = idx
      selected = obj.inputs{i}.findRecursive(what, n, depth - 1, visited, selected) ;
    end
  end
  
  % mark as seen
  visited(obj.id) = true ;
  
  % mark self as selected, if it matches the pattern
  sel = false ;
  if ischar(what)
    if any(what == '*') || any(what == '?')  % wildcards
      if ~isempty(regexp(obj.name, regexptranslate('wildcard', what), 'once'))
        sel = true ;
      end
    elseif isequal(obj.name, what) || isa(obj, what)
      sel = true ;
    end
  elseif isempty(what) || isequal(obj.func, what)
    sel = true ;
  end
  
  if sel
    selected{end+1} = obj ;
  end
  
end
