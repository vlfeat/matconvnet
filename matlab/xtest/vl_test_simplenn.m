function test_tinynet()

h = 244 ;
w = 244 ;
n = 3 ;

%
% Define a CNN
%
net.layers = {} ;

% conv layer 1
net.layers{end+1} = struct('type','conv', ...
                           'filters', randn(3,3,3,15,'single'), ...
                           'biases', randn(15,1,'single'), ...
                           'stride', 1, 'pad', 0) ;
net.layers{end+1} = struct('type','pool', ...
                           'pool', [3 3], ...
                           'stride', 1, 'pad', 0) ;
net.layers{end+1} = struct('type','relu') ;
net.layers{end+1} = struct('type','normalize', 'param', [5 .1 .01 .75]) ;

% conf layer 2
net.layers{end+1} = struct('type','conv', ...
                           'filters', randn(3,3,15,30,'single'), ...
                           'biases', randn(30,1,'single'), ...
                           'stride', 1, 'pad', 0) ;
net.layers{end+1} = struct('type','pool', ...
                           'pool', [3 3], ...
                           'stride', 1, 'pad', 0) ;
net.layers{end+1} = struct('type','relu') ;
net.layers{end+1} = struct('type','normalize', 'param', [5 .1 .01 .75]) ;

% fully connected layers
net.layers{end+1} = struct('type','conv', 'filters', randn(h-8, w-8, 30, 64, 'single'), 'biases', randn(1,64,'single'), ...
  'stride', 1, 'pad', 0) ;
net.layers{end+1} = struct('type','relu') ;
net.layers{end+1} = struct('type','conv', 'filters', randn(1,1, 64, 64, 'single'), 'biases', randn(1,64,'single'), ...
  'stride', 1, 'pad', 0) ;
net.layers{end+1} = struct('type','relu') ;
net.layers{end+1} = struct('type','conv', 'filters', randn(1,1, 64, 32, 'single'), 'biases', randn(1,32,'single'), ...
  'stride', 1, 'pad', 0) ;

% softmax and loss
net.layers{end+1} = struct('type', 'softmax') ;

%net.layers{end+1} = struct('type', 'loss', 'class', 12) ;

%
% Test algorithms
%

im = randn(h,w,3,n,'single') ;

% forward-backward passes
dzdy = randn(1,1,32,n)  ;
for i=1:n
  res_{i} = vl_simplenn(net, im(:,:,:,i), dzdy(:,:,:,i)) ;
end
res = vl_simplenn(net, im, dzdy) ;

for i=1:numel(res)
  if i==numel(res)
    name = 'output' ;
  else
    name = net.layers{i}.type ;
  end
  x = res(i).x ;
  dx = res(i).dzdx ;
  dw = res(i).dzdw ;
  if ~isempty(dw)
    for k=1:2
      dw_ = zeros('like',dw{k}) ;
      for j = 1:numel(res_)
        dw_ = dw_ + res_{j}(i).dzdw{k} ;
      end
      vl_testsim(dw_, dw{k}) ;
    end
  end
  for j=1:numel(res_)
    x_ = res_{j}(i).x ;
    dx_ = res_{j}(i).dzdx ;    
    vl_testsim(x_, x(:,:,:,j)) ;
    vl_testsim(dx_, dx(:,:,:,j)) ;
    fprintf('layer (%2d,%d,%10s) ok\n', i,j,name) ;
  end
end
