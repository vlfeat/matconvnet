function mask = vl_nnmask(x, rate, dy)

if nargin == 3
  % backward mode; the derivative of x is expected. since we only need x to
  % get its size, it doesn't change, so its derivative is the identity.
  mask = dy ;
  return
end

if nargin < 2
  rate = 0.5 ;
end

scale = 1 / (1 - rate) ;

if isa(x,'gpuArray')
  switch classUnderlying(x)
    case 'single'
      scale = single(scale) ;
    case 'double'
      scale = double(scale) ;
  end
  mask = scale * (gpuArray.rand(size(x), 'single') >= rate) ;
else
  switch class(x)
    case 'single'
      scale = single(scale) ;
    case 'double'
      scale = double(scale) ;
  end
  mask = scale * (rand(size(x), 'single') >= rate) ;
end
