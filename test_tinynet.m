function test_tinynet()

h = 244 ;
w = 244 ;

%
% Define a CNN
%
net.layers = {} ;

% conv layer 1
net.layers{end+1} = struct('type','conv', 'w', randn(3,3,3,15,'single')) ;
net.layers{end+1} = struct('type','pool', 'pool', [3 3]) ;
net.layers{end+1} = struct('type','relu') ;
net.layers{end+1} = struct('type','normalize', 'param', [5 .1 .01 .75]) ;

% conf layer 2
net.layers{end+1} = struct('type','conv', 'w', randn(3,3,15,30,'single')) ;
net.layers{end+1} = struct('type','pool', 'pool', [3 3]) ;
net.layers{end+1} = struct('type','relu') ;
net.layers{end+1} = struct('type','normalize', 'param', [5 .1 .01 .75]) ;

% fully connected layers
net.layers{end+1} = struct('type','vec') ;
net.layers{end+1} = struct('type','fully', 'w', randn(64, (h-8)*(w-8)*30)) ;
net.layers{end+1} = struct('type','relu') ;
net.layers{end+1} = struct('type','fully', 'w', randn(64, 64)) ;
net.layers{end+1} = struct('type','relu') ;
net.layers{end+1} = struct('type','fully', 'w', randn(32, 64)) ;

% softmax and loss
net.layers{end+1} = struct('type', 'softmax') ;
net.layers{end+1} = struct('type', 'loss', 'class', 12) ;

%
% Test algorithms
%

% forward pass
im = randn(h,w,3,'single') ;
res = tinynet(net, im) ;

% backward pass
dzdy = 1 ;
res_ = tinynet(net, im, dzdy) ;

keyboard
