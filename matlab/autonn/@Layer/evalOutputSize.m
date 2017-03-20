function sz = evalOutputSize(obj, varargin)
% SZ = OBJ.EVALOUTPUTSIZE('INPUT1', SZ1, 'INPUT2', SZ2, ...)
% Computes the output size of a Layer, given input names and respective
% sizes. Note that the size is computed by compiling and evaluating a
% network, which always outputs reliable sizes but can be computationally
% expensive.
%
% SZ = OBJ.EVALOUTPUTSIZE(LAYERS, 'INPUT1', SZ1, 'INPUT2', SZ2, ...)
% Evaluates output sizes of multiple Layers, given in cell array LAYERS.
% The corresponding outputs sizes are also returned in a cell array.
% Note that they must be part of the computational graph of OBJ.

% NOTE: The upside of compiling a dummy network is that there's no need to
% specify size calculations for all layers by hand. This is especially
% useful for many Matlab native functions, and flexible enough for
% user-defined custom layers to be supported with no extra work.

% Copyright (C) 2016 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  if ~isempty(varargin) && iscell(varargin{1})
    layers = varargin{1} ;  % multiple layers given
    varargin(1) = [] ;
  else
    layers = {obj} ;  % single layer
  end

  assert(iscellstr(varargin(1:2:end)), 'Expected a list of input names and their sizes.') ;

  inputNames = cellfun(@(o) o.name, obj.find('Input'), 'UniformOutput',false) ;
  inputNames(strcmp(inputNames, 'testMode')) = [] ;  % these inputs are set automatically and are not needed
  inputs = cell(1, 2 * numel(inputNames)) ;

  for i = 1:numel(inputNames)
    % find the user-supplied name that matches this network input
    match = find(strcmp(inputNames{i}, varargin(1:2:end))) ;
    assert(~isempty(match), ['Missing size for input ''' inputNames{i} '''.']) ;
    
    % add it to the list, along with its initial value
    inputSz = varargin{2 * match} ;
    inputs{2 * i} = zeros(inputSz, 'single') ;
    inputs{2 * i - 1} = inputNames{i} ;
  end
  
  % note any user-supplied names that do not exist in the network yet are
  % silently ignored (i.e., when building a network, and that part hasn't
  % been defined, or hasn't been connected to this Layer yet).
  
  % compile and evaluate network
  net = Net(obj, 'sequentialNames',false, 'shortCircuit',false, 'forwardOnly',true) ;
  net.setInputs(inputs{:}) ;
  net.eval('forward') ;
  
  % retrieve values of given layers, and return their sizes
  sz = cell(size(layers)) ;
  for i = 1:numel(layers)
    sz{i} = size(net.getValue(layers{i})) ;
  end
  
  if isscalar(sz)
    sz = sz{1} ;
  end

end

