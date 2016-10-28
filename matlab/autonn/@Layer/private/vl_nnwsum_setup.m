function inputs = vl_nnwsum_setup(layer)
%VL_NNWSUM_SETUP
%   Setup a weighted sum layer, by merging any other weighted sums in its
%   inputs.
%   Called by AUTONN_SETUP.

% Copyright (C) 2016 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  assert(isequal(layer.func, @vl_nnwsum)) ;
  
  % make sure there's a 'weights' property at the end, with the correct size
  assert(strcmp(layer.inputs{end-1}, 'weights') && ...
    numel(layer.inputs{end}) == numel(layer.inputs) - 2) ;

  % separate inputs to the sum, and weights
  inputs = layer.inputs(1:end-2) ;
  origWeights = layer.inputs{end} ;
  weights = cell(size(inputs)) ;

  for k = 1 : numel(inputs)
    in = inputs{k} ;
    if isa(in, 'Layer') && isequal(in.func, @vl_nnwsum) && in.optimize
      % merge weights and store results
      inputs{k} = in.inputs(1:end-2) ;
      weights{k} = origWeights(k) * in.inputs{end} ;
    else
      % any other input (Layer or constant), wrap it in a single cell
      inputs{k} = {in} ;
      weights{k} = origWeights(k) ;
    end
  end

  % merge the results in order
  inputs = [inputs{:}] ;
  weights = [weights{:}] ;

  % append weights as a property
  inputs = [inputs, {'weights', weights}] ;
end

