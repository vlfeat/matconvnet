function inputs = vl_nnconv_setup(layer)
%VL_NNCONV_SETUP
%   Setup a conv or convt layer: if there is a 'size' argument,
%   automatically initialize randomized Params for the filters. Also
%   handles 'hasBias' (initialize biases), 'learningRate' and 'weightDecay'
%   arguments for the Params.

  assert(isequal(layer.func, @vl_nnconv) || isequal(layer.func, @vl_nnconvt)) ;
  
  inputs = layer.inputs ;
  
  % parse options. other options such as 'stride' will be maintained in the
  % inputs list.
  opts = struct('size', [], 'hasBias', false, 'learningRate', 1, 'weightDecay', 1) ;
  
  [opts, inputs] = vl_argparsepos(opts, inputs) ;
  
  if ~isempty(opts.size)
    % a size was specified, create Params
    scale = sqrt(2 / prod(opts.size(1:3))) ;
    filters = Param('value', randn(opts.size, 'single') * scale, ...
                    'learningRate', opts.learningRate(1), ...
                    'weightDecay', opts.weightDecay(1)) ;

    if opts.hasBias
      % also create bias
      biases = Param('value', zeros(sz(4), 1, 'single') * scale, ...
                     'learningRate', opts.learningRate(max(1,end)), ...
                     'weightDecay', opts.weightDecay(max(1,end))) ;
    else
      biases = [] ;
    end

    %include filters and biases in inputs list
    inputs = [inputs(1), {filters, biases}, inputs(2:end)] ;
  end 
end
