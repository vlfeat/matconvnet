function vl_nnloss_setup(layer)
%VL_NNLOSS_SETUP
%   Setup loss, by specifying that the 2nd input (label) has no derivative.

  layer.numInputDer = 1 ;

end

