function inputs = vl_nnconv_setup(layer)
%VL_NNCONV_SETUP
%   Setup a conv or convt layer: if there is a 'size' argument,
%   automatically initialize randomized Params for the filters and
%   biases.

  assert(isequal(layer.func, @vl_nnconv) || isequal(layer.func, @vl_nnconvt)) ;
  
  inputs = layer.inputs ;

  % parse 'size' argument
  szPos = find(strcmp(inputs, 'size'), 1) ;
  
  if ~isempty(szPos)
    sz = inputs{szPos + 1} ;
    scale = sqrt(2 / prod(sz(1:3))) ;
    filters = Param('value', randn(sz, 'single') * scale) ;

    inputs(szPos : szPos + 1) = [] ;  % eliminate 'size' argument

    % parse 'hasBias' argument
    biasPos = find(strcmp(inputs, 'hasBias'), 1) ;
    biases = [] ;
    if isempty(biasPos) || ~inputs{szPos + 1}
      biases = Param('value', zeros(sz(4), 1, 'single') * scale) ;
      inputs(biasPos : biasPos + 1) = [] ;  % eliminate 'hasBias' argument
    end

    inputs = [inputs(1), {filters, biases}, inputs(2:end)] ;
  end 
end

