function inputs = vl_nnbinaryop_setup(layer)
%VL_NNBINARYOP_SETUP
%   Setup binaryop: @ldivide is just @rdivide with swapped inputs.

  assert(isequal(layer.func, @vl_nnbinaryop)) ;
  
  inputs = layer.inputs ;
  
  if isequal(inputs{3}, @ldivide)
    inputs = [inputs([2,1]), {@rdivide}] ;
  end

end

