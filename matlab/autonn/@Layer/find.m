function objs = find(obj, varargin)
% OBJS = OBJ.FIND()
% OBJS = OBJ.FIND(NAME/FUNC/CLASS)
% Finds layers, starting at the given output layer. The search
% criteria can be a layer name, a function handle, or a class name
% (such as 'Input' or 'Param').
% By default a cell array is returned, which may be empty.
%
% OBJS = OBJ.FIND(..., N)
% Returns only the Nth object that fits the criteria, in the order of
% a forward pass (e.g. from the first layer). If N is negative, it is
% found in the order of a backward pass (e.g. from the last layer,
% which corresponds to N = -1).
% Raises an error if no object is found.
%
% OBJS = OBJ.FIND(..., 'depth', D)
% Only recurses D depth levels (i.e., D=1 means that only OBJ's
% inputs will be searched).

% Copyright (C) 2016 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).


  % parse name-value pairs, and leave the rest in varargin
  opts.depth = inf ;
  firstArg = find(cellfun(@(s) ischar(s) && any(strcmp(s, fieldnames(opts))), varargin), 1) ;
  if ~isempty(firstArg)
    opts = vl_argparse(opts, varargin(firstArg:end), 'nonrecursive') ;
    varargin(firstArg:end) = [] ;
  end

  what = [] ;
  n = 0 ;
  if isscalar(varargin)
    if isnumeric(varargin{1})
      n = varargin{1} ;
    else
      what = varargin{1} ;
    end
  elseif numel(varargin) == 2
    what = varargin{1} ;
    n = varargin{2} ;
  elseif numel(varargin) > 3
    error('Too many input arguments.') ;
  end

  % do the work
  objs = findRecursive(obj, what, n, opts.depth, Layer.initializeRecursion(), {}) ;

  % choose the Nth object
  if n ~= 0
    assert(numel(objs) >= abs(n), 'Cannot find a layer fitting the specified criteria.')
    if n > 0
      objs = objs{n} ;
    else
      objs = objs{numel(objs) + n + 1} ;
    end
  end
end

function selected = findRecursive(obj, what, n, depth, visited, selected)
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
      selected = findRecursive(obj.inputs{i}, what, n, depth - 1, visited, selected) ;
    end
  end
  
  % add self to selected list, if it matches the pattern
  if ~visited.isKey(obj.id)  % not in the list yet
    if ischar(what)
      if any(what == '*') || any(what == '?')  % wildcards
        if ~isempty(regexp(obj.name, regexptranslate('wildcard', what), 'once'))
          selected{end+1} = obj ;
        end
      elseif isequal(obj.name, what) || isa(obj, what)
        selected{end+1} = obj ;
      end
    elseif isempty(what) || isequal(obj.func, what)
      selected{end+1} = obj ;
    end
  end
  
  % mark as seen
  visited(obj.id) = true ;
  
end

