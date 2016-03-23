function vl_nnsoftmaxloss_setup(layer)
%VL_NNSOFTMAXLOSS_SETUP
%   Setup loss, by specifying that the 2nd input (label) has no derivative.

  layer.numInputDer = 1 ;

end

