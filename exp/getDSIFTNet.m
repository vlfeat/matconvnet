function net = getDSIFTNet(binSize)
% Define a CNN roughly equivalent to dense SIFT

if nargin < 1
  binSize = 5 ;
end

% spatial derivatives along 8 directions
dx = [0 0 0 ; -1 0 1 ; 0  0 0]/2 ;
dy = dx' ;
c = cos(pi/4) ;
spatialDer = single(cat(4, ...
  dx, c*dx+c*dy, dy, -c*dx+c*dy, -dx, -c*dx-c*dy, -dy, c*dx-c*dy)) ;

% the norm of the ouput of the 8 derivative filter is twice the norm
% of the gradient, exactly
if 0
  for t =1:100
    x = randn(3) ;
    g = reshape(spatialDer, [], 8)'*x(:) ;
    gx = (x(2,3) - x(2,1))/2 ;
    gy = (x(3,2) - x(1,2))/2 ;
    a(t) = norm([gx gy]);
    b(t) = norm([g]) ; %twice the norm of a
  end
end

% orientation binning is the weakest approximation. It could be done much
% better with a specialised network layer. A good way of doing it is to
% take the directional derivatives normalised by the gradient modulus; this
% gives the cosine of the angle between that derivative and the gradient.
% Scaling and biasing the latter appropritely + max non linearity is a
% pretty good approximation of smooth binning. Here we simply do max with
% zero.

% spatial bilinear windowing for bins; use an acrane functionality of gconv to apply
% the same filter to each spatial derivative
a = linspace(0,1,binSize) ;
a = [a a(end-1:-1:1)] ;
bilinearFilter = repmat(single(a'*a), [1, 1, 1, 8]) ;

% stacking of spatial bins into 128 dim. descriptors
sigma = 1.5 ;
mask = {} ;
t = 0 ;
for i=1:4
  for j=1:4
    for o=1:8
      t=t+1 ;
      mask{t} = zeros(4,4,8) ;
      mask{t}(i,j,o) = exp(-0.5*((i-2.5).^2 + (j-2.5).^2) / sigma^2) ;
    end
  end
end
mask = single(cat(4, mask{:})) ;

net.layers = {} ;
net.layers{end+1} = struct('type','conv', ...
  'filters', spatialDer, ...
  'biases', zeros(size(spatialDer,4),1,'single'), ...
  'stride', 1, 'pad', 0) ;
%net.layers{end+1} = struct('type','noffset', 'param', [0.5*cos(pi/4) .5]) ;
net.layers{end+1} = struct('type','relu') ;
net.layers{end+1} = struct('type','conv', ...
  'filters', bilinearFilter, ...
  'biases', zeros(size(bilinearFilter,4),1,'single'), ...
  'stride', binSize, 'pad', 0) ;
net.layers{end+1} = struct('type','conv', ...
  'filters', mask, ...
  'biases', zeros(size(mask,4),1,'single'), ...
  'stride', 3, 'pad', 0) ;
net.layers{end+1} = struct('type','normalize', 'param', [128*2, 0.0001, 1, .5]) ;