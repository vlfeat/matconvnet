function sequentialNames(varargin)
% OBJ.SEQUENTIALNAMES()
% Sets layer names sequentially, based on their function handle and
% execution order. E.g.: conv1, conv2, pool1...
% Only empty names are set.
%
% LAYER.SEQUENTIALNAMES(OBJ1, OBJ2, ...)
% Same but considering a network with outputs OBJ1, OBJ2, ...
%
% LAYER.SEQUENTIALNAMES(..., MODIFIER)
% Specifies a function handle to be evaluated on each name, possibly
% modifying it (e.g. append a prefix or suffix).
%
% See also WORKSPACENAMES.

% Copyright (C) 2016 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).


  if ~isempty(varargin) && isa(varargin{end}, 'function_handle')
    modifier = varargin{end} ;
    varargin(end) = [] ;
  else
    modifier = @deal ;
  end

  % list all layer objects, in forward order
  assert(~isempty(varargin), 'Not enough input arguments.') ;
  objs = cell(1, numel(varargin)) ;
  for i = 1:numel(varargin)
    objs{i} = varargin{i}.find() ;
  end
  objs = [objs{:}] ;  % flatten array of results


  % automatically set missing names, according to the execution order
  for k = 1:numel(objs)
    if isa(objs{k}, 'Input')  % input1, input2, ...
      if isempty(objs{k}.name)
        n = nnz(cellfun(@(o) isa(o, 'Input'), objs(1:k))) ;
        objs{k}.name = modifier(sprintf('input%i', n)) ;  %#ok<*AGROW>
      end
    elseif ~isempty(objs{k}.func)  % conv1, conv2...
      name = objs{k}.name ;
      if isempty(name)
        n = nnz(cellfun(@(o) isequal(o.func, objs{k}.func), objs(1:k))) ;
        name = func2str(objs{k}.func) ;  %#ok<*PROP>

        if strncmp(name, 'vl_nn', 5), name(1:5) = [] ; end  % shorten names
        if strcmp(name, 'bnorm_wrapper'), name = 'bnorm' ; end
        name = sprintf('%s%i', name, n) ;
        objs{k}.name = modifier(name) ;
      end

      % set dependant Param names for vl_nnconv and vl_nnbnorm
      suffix = [] ;
      if isequal(objs{k}.func, @vl_nnconv)
        suffix = {[], 'filters', 'biases'} ;
      elseif isequal(objs{k}.func, @vl_nnbnorm_wrapper)
        suffix = {[], 'g', 'b', 'moments'} ;
      end
      inputs = objs{k}.inputs ;
      if ~isempty(suffix)
        for i = 2 : min(numel(suffix), numel(inputs))
          if isa(inputs{i}, 'Param') && isempty(inputs{i}.name)
            inputs{i}.name = modifier(sprintf('%s_%s', name, suffix{i})) ;
          end
        end
      end
      
      % set dependant Param names for general layers: name_p1, name_p2, ...
      ps = find(cellfun(@(o) isa(o, 'Param'), inputs)) ;
      for i = 1:numel(ps)
        if isempty(inputs{ps(i)}.name)
          inputs{ps(i)}.name = modifier(sprintf('%s_p%i', name, i)) ;
        end
      end
    end
  end

  % name any remaining objects by class
  for k = 1:numel(objs)
    if isempty(objs{k}.name)
      n = nnz(cellfun(@(o) isa(o, class(objs{k})), objs(1:k))) ;
      objs{k}.name = modifier(sprintf('%s%i', lower(class(objs{k})), n)) ;
    end
  end
end
