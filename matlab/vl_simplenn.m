function res = tinynet(net, x, dzdy)
doder = (nargin >= 3) ;

n = numel(net.layers) ;

res = struct(...
  'x', cell(1,n+1), ...
  'dzdx', cell(1,n+1), ...
  'dzdw', cell(1,n+1), ...
  'time', num2cell(zeros(1,n+1)), ...
  'backwardTime', num2cell(zeros(1,n+1))) ;
res(1).x = x ;

for i=1:n
  l = net.layers{i} ;
  res(i).time = tic ;
  switch l.type
    case 'conv'
      res(i+1).x = vl_nnconv(res(i).x, l.filters, l.biases, 'pad', l.pad, 'stride', l.stride) ;
    case 'pool'
      res(i+1).x = vl_nnpool(res(i).x, l.pool, 'pad', l.pad, 'stride', l.stride) ;
    case 'normalize'
      res(i+1).x = vl_nnnormalize(res(i).x, l.param) ;
    case 'fully'
      res(i+1).x = vl_nnfull(res(i).x, l.w, l.b) ;
    case 'softmax'
      res(i+1).x = vl_nnsoftmax(res(i).x) ;
    case 'loss'
      res(i+1).x = vl_nnloss(res(i).x, l.class) ;
    case 'vec'
      res(i+1).x = vl_nnvec(res(i).x) ;
    case 'relu'
      res(i+1).x = vl_nnrelu(res(i).x) ;
    case 'noffset'
      res(i+1).x = vl_nnnoffset(res(i).x, l.param) ;
    otherwise
      error('Unknown layer type %s', l.type);
  end
  try
    wait(gpuDevice) ;
  catch
    % no gpuDevice
  end
  res(i).time = toc(res(i).time) ;
end

if doder
  res(n+1).dzdx = dzdy ;
  for i=n:-1:1
    l = net.layers{i} ;
    res(i).backwardTime = tic ;
    switch l.type
      case 'conv'
        [dzdx, dzdw, dzdb] = vl_nnconv(res(i).x, l.filters, l.biases, ...
          res(i+1).dzdx, ...
          'pad', l.pad, 'stride', l.stride) ;
        res(i).dzdx = dzdx;
        res(i).dzdw = {dzdw,dzdb} ;
        clear dzdx dzdw dzdb ;
      case 'pool'
        res(i).dzdx = vl_nnpool(res(i).x, l.pool, res(i+1).dzdx, ...
          'pad', l.pad, 'stride', l.stride) ;
      case 'normalize'
        res(i).dzdx = vl_nnnormalize(res(i).x, l.param, res(i+1).dzdx) ;
      case 'fully'
        [dzdx, dzdw, dzdb] = vl_nnfull(res(i).x, l.w, l.b, res(i+1).dzdx) ;
        res(i).dzdx = dzdx ;
        res(i).dzdw = {dzdw, dzdb} ;
        clear dzdx dzdw dzdb ;
      case 'softmax'
        res(i).dzdx = vl_nnsoftmax(res(i).x, res(i+1).dzdx) ;
      case 'loss'
        res(i).dzdx = vl_nnloss(res(i).x, l.class, res(i+1).dzdx) ;
      case 'vec'
        res(i).dzdx = vl_nnvec(res(i).x, res(i+1).dzdx) ;
      case 'relu'
        res(i).dzdx = vl_nnrelu(res(i).x, res(i+1).dzdx) ;
      case 'noffset'
        res(i).dzdx = vl_nnoffset(res(i).x, l.param, res(i+1).dzdx) ;
    end
    try
        wait(gpuDevice) ;
    catch
        % no gpuDevice
    end
    res(i).backwardTime = toc(res(i).backwardTime) ;
  end
end
