function inputs = vl_nnmatrixop_setup(layer)
%VL_NNMATRIXOP_SETUP
%   Setup matrixop: @mldivide is just @mrdivide with swapped inputs.

  assert(isequal(layer.func, @vl_nnmatrixop)) ;
  
  inputs = layer.inputs ;
  
  if isequal(inputs{3}, @mldivide)
    inputs = [inputs([2,1]), {@mrdivide}] ;
  end

end

