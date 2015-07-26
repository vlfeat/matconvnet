function move(obj, device)
% MOVE  Move the DagNN to either CPU or GPU
%   
%
obj.reset() ;
obj.device = device ;
switch device
  case 'gpu'
    for i=1:numel(obj.params)
      obj.params(i).value = gpuArray(obj.params(i).value) ;
    end
  case 'cpu'
    for i=1:numel(obj.params)
      obj.params(i).value = gather(obj.params(i).value) ;
    end
  otherwise
    error('DEVICE must be either ''cpu'' or ''gpu''.') ;
end
for l = 1:numel(obj.layers)
  obj.layers(l).block.move(device) ;
end
