function optimizeVars(net, opts, objs)
%OPTIMIZEVARS
%   Variable allocation optimizations, such as ReLU short-circuiting.
%   Called by BUILD.
%
%   Assumes BUILDORDER was called, deciding the execution order (OBJS
%   contains the ordered layers), and assigning all needed output vars
%   (OBJS{K}.OUTPUTVAR). This function can change the assigned output vars.

% Copyright (C) 2016 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).


  % Short-circuiting a ReLU amounts to setting the outputVar to be the same
  % as the input var. The corresponding derivative also gets reused. This
  % requires that the result of the backward ReLU *replace* the derivative,
  % not be accumulated as usual (layer.accumDer = false).
  %
  % It cannot be done when:
  % 1) The same input var is used by other layers.
  % 2) The input is a Param; otherwise the value and its derivative would
  %    get overwritten. It would also cause trouble with sub-batches
  %    (Param derivative accumulation).
  %
  % An alternative in case 1 would be to delete the output var after it is
  % used by the next layers, but this is not implemented currently.
  
  if opts.shortCircuit
    % cell array of cell arrays with each layer's input layers
    objInputs = cell(numel(objs), 1) ;
    isLayer = @(o) isa(o, 'Layer') ;
    for k = 1:numel(objs)
      objInputs{k} = objs{k}.inputs(cellfun(isLayer, objs{k}.inputs)) ;
    end
    
    for k = 1:numel(objs)
      % a ReLU with Layer (non-constant) input, that is not a Param
      if isequal(objs{k}.func, @vl_nnrelu) && ~isempty(objs{k}.inputs) && ...
       isa(objs{k}.inputs{1}, 'Layer') && ~isa(objs{k}.inputs{1}, 'Param')

        % check if any other layer is using the same input
        otherInputs = [objInputs{[1:k-1, k+1:end]}] ;  % flatten cell arrays into one
        x = objs{k}.inputs{1} ;
        if ~any(cellfun(@(o) eq(o, x, 'sameInstance'), otherInputs))
          
          % it's safe, short-circuit it
          objs{k}.outputVar = objs{k}.inputs{1}.outputVar ;
          objs{k}.accumDer = false ;
        end
      end
    end
  end
  
end

