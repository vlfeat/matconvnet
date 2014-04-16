function net = tinynet_move(net, destination)
switch destination
  case 'gpu', moveop = @(x) gpuArray(x) ;
  case 'cpu', moveop = @(x) gather(x) ;
  otherwise, error('Unknown desitation ''%s''.', destination) ;
end
for l=1:numel(net.layers)
  switch net.layers{l}.type
    case 'conv'
      net.layers{l}.filters = moveop(net.layers{l}.filters) ;
      net.layers{l}.biases = moveop(net.layers{l}.biases) ;
    case 'fully'
      net.layers{l}.w = moveop(net.layers{l}.w) ;
      net.layers{l}.b = moveop(net.layers{l}.b) ;
    otherwise
      % nothing to do
  end
end
