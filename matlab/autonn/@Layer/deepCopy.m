function other = deepCopy(obj, varargin)
% OTHER = OBJ.DEEPCOPY(SHAREDLAYER1, SHAREDLAYER2, ...)
% OTHER = OBJ.DEEPCOPY({SHAREDLAYER1, SHAREDLAYER2, ...})
% Returns a deep copy of a layer, excluding SHAREDLAYER1,
% SHAREDLAYER2, etc, which are optional. This can be used to
% implement shared Params, or define the boundaries of the deep copy.
%
% OTHER = OBJ.DEEPCOPY(..., RENAME)
% Specifies a function handle to be evaluated on each name, possibly
% modifying it (e.g. append a prefix or suffix).
%
% OTHER = OBJ.DEEPCOPY(..., 'noName')
% Does not copy object names (they are left empty).
%
% To create a shallow copy, use OTHER = OBJ.COPY().

  shared = varargin ;  % list of shared layers
  rename = @deal ;  % no renaming by default
  if ~isempty(shared)
    if isa(shared{end}, 'function_handle')
      rename = shared{end} ;  % specified rename function
      shared(end) = [] ;
    elseif ischar(shared{end})
      assert(strcmp(shared{end}, 'noName'), 'Invalid option.') ;
      rename = @(~) [] ;  % assign empty to name
      shared(end) = [] ;
    end
  end

  if isscalar(shared) && iscell(shared{1})  % passed in cell array
    shared = shared{1} ;
  end
  
  % map between (original) object ID and its copied instance. also acts as
  % a 'visited' list, to avoid redundant recursions.
  visited = Layer.initializeRecursion() ;
  
  for i = 1:numel(shared)
    assert(~eq(shared{i}, obj, 'sameInstance'), 'The root layer of a deep copy cannot be a shared layer.') ;
    
    % propagate shared status (see below)
    shareRecursive(shared{i}, visited) ;
  end

  % do the actual copy
  other = deepCopyRecursive(obj, rename, visited) ;
end

function shareRecursive(shared, visited)
  % shared layers are just considered visited/copied, pointing to
  % themselves as the new copy.
  visited(shared.id) = shared ;
  
  % propagate shared status: any layer that this layer depends on must also
  % be shared. otherwise, a shared layer would be depending on both the
  % original layer and its copy; a contradiction that leads to subtle bugs.
  for i = 1:numel(shared.inputs)
    in = shared.inputs{i} ;
    if isa(in, 'Layer') && ~visited.isKey(in.id)
      shareRecursive(in, visited) ;
    end
  end
end

function other = deepCopyRecursive(original, rename, visited)
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
        other.inputs{i} = deepCopyRecursive(in, rename, visited) ;
      end
      
      in.enableCycleChecks = true ;
    end
  end
end

