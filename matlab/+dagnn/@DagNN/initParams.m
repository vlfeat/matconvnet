function initParams(obj)
% INITPARAM  Initialize the paramers of the DagNN
%   OBJ.INITPARAM() uses the INIT() method of each layer to initialize
%   the corresponding parameters (usually randomly).

for l = 1:numel(obj.layers)
  p = obj.getParamIndex(obj.layers(l).params) ;
  params = obj.layers(l).block.init() ;
  switch obj.device
    case 'cpu'
      params = cellfun(@gather, params, 'UniformOutput', false) ;
    case 'gpu'
      params = cellfun(@gpuArray, params, 'UniformOutput', false) ;
  end
  [obj.params(p).value] = deal(params{:}) ;
end
