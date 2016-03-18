function inputs = vl_nnwsum_setup(layer)
%VL_NNWSUM_SETUP
%   Setup a weighted sum layer, by merging any other weighted sums in its
%   inputs.

  assert(isequal(layer.func, @vl_nnwsum)) ;
  
  % make sure there's a 'weights' property at the end, with the correct size
  assert(strcmp(layer.inputs{end-1}, 'weights') && ...
    numel(layer.inputs{end}) == numel(layer.inputs) - 2) ;

  weights = layer.inputs{end}(:)' ;
  valid = true(1, numel(layer.inputs) - 2) ;
  merged_inputs = {} ;
  merged_weights = [] ;

  % iterate all inputs except for the 'weights' property
  for i = 1 : numel(layer.inputs) - 2
    in = layer.inputs{i} ;
    if isa(in, 'Layer') && isequal(in.func, @vl_nnwsum)
      merged_inputs = [merged_inputs, in.inputs(1:end-2)] ;  %#ok<*AGROW>
      merged_weights = [merged_weights, weights(i) * in.inputs{end}] ;
      valid(i) = false ;
    end
  end

  weights = [weights(valid), merged_weights] ;
  inputs = [layer.inputs(valid), merged_inputs, {'weights', weights}] ;
end

