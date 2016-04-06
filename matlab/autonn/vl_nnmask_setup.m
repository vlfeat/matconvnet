function vl_nnmask_setup(layer)
%VL_NNMASK_SETUP
%   Setup a dropout mask generator layer.

  assert(isequal(layer.func, @vl_nnmask)) ;
  
  % remove layer in test mode
  layer.testFunc = 'none' ;

end

