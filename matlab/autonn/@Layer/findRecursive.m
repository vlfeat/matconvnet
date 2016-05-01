function objs = findRecursive(obj, what, n, depth, objs)
% FINDRECURSIVE Recursion on layers, used by FIND.

% Copyright (C) 2016 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).


  if n > 0 && numel(objs) >= n
    return  % early break, already found the object we're after
  end

  % recurse on inputs not on the list yet (forward order)
  if depth > 0
    for i = 1:numel(obj.inputs)
      if isa(obj.inputs{i}, 'Layer') && ~any(cellfun(@(o) isequal(obj.inputs{i}, o), objs))
        objs = obj.inputs{i}.findRecursive(what, n, depth - 1, objs) ;
      end
    end
  end

  % add self to list if it matches the pattern
  if ischar(what)
    if any(what == '*') || any(what == '?')  % wildcards
      if ~isempty(regexp(obj.name, regexptranslate('wildcard', what), 'once'))
        objs{end+1} = obj ;
      end
    elseif isequal(obj.name, what) || isa(obj, what)
      objs{end+1} = obj ;
    end
  elseif isequal(obj.func, what)
    objs{end+1} = obj ;
  end
end
