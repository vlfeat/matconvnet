function exp_sift_like()

run(fullfile(mfilename('fullpath'), '../../matlab/vlg_setup.m')) ;
run ~/src/vlfeat/toolbox/vl_setup ;

h = 128 ;
w = 128 ;

% -------------------------------------------------------------------------
% Define a CNN
% -------------------------------------------------------------------------

fid = fopen('data/chars.txt','r','n','UTF-8') ;
chars=[];
while ~feof(fid), chars = [chars, fgetl(fid)] ; end  
fclose(fid) ;

im = vl_impattern('river1') ;
im = rgb2gray(im2single(imresize(im, .25))) ;

% -------------------------------------------------------------------------
% Define a CNN roughly simulating SIFT
% -------------------------------------------------------------------------

binSize = 5 ;

% spatial derivatives along 8 directions
dx = [0 0 0 ; -1 0 1 ; 0  0 0] ;
dy = dx' ;
c = cos(pi/4) ;
spatialDer = single(cat(4, ...
  dx, c*dx+c*dy, dy, -c*dx+c*dy, -dx, -c*dx-c*dy, -dy, c*dx-c*dy)) ;

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
net.layers{end+1} = struct('type','relu') ;
net.layers{end+1} = struct('type','conv', ...
  'filters', bilinearFilter, ...
  'biases', zeros(size(bilinearFilter,4),1,'single'), ...
  'stride', binSize, 'pad', 0) ;
net.layers{end+1} = struct('type','conv', ...
  'filters', mask, ...
  'biases', zeros(size(mask,4),1,'single'), ...
  'stride', binSize, 'pad', 0) ;
net.layers{end+1} = struct('type','normalize', 'param', [128*2, 0.0001, 1, .5]) ; 

res = tinynet(net, im) ;

descrs_ = reshape(res(end-1).x,[],128)' ;
descrs = reshape(res(end).x,[],128)' ;

figure(1) ; clf ;
vl_imarraysc(res(end).x) ;

% now the derivative w.r.t a cost function is really easy
dzdy = randn(size(res(end).x), 'single') ;
res_ = tinynet(net, im, dzdy) ;



