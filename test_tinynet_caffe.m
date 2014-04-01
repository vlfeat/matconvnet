if ~exist('net')
  net = load('tinynet_caffe.mat') ;
  load('tinynet_caffe_data.mat') ;
end

% patch
net.layers{4}.param = [5 1 1e-4/5 0.75] ;
net.layers{8}.param = [5 1 1e-4/5 0.75] ;

% forward pass
res = tinynet(net, im) ;

% backward pass
dzdy = 1 ;
res_ = tinynet(net, im, dzdy) ;

% Block 0 (input)
assert(max(res(1).x(:)-data{1}(:)) < 1e-3) ; % input image

% Block 1
assert(max(res(2).x(:)-data{2}(:)) < 1e-3) ; % output conv
assert(max(res(4).x(:)-data{3}(:)) < 1e-3) ; % otuput relu+pool
assert(max(res(5).x(:)-data{4}(:)) < 1e-3) ; % output norm

% Block 2
assert(max(res(6).x(:)-data{6}(:)) < 1e-3) ; % output conv
assert(max(res(8).x(:)-data{7}(:)) < 1e-3) ; % output relu+pool
assert(max(res(9).x(:)-data{8}(:)) < 1e-3) ; % output norm

% Block 3
assert(max(res(10).x(:)-data{10}(:)) < 1e-3) ; % output conv

% Block 4
assert(max(res(12).x(:)-data{12}(:)) < 1e-3) ; % output relu+conv

% Block 5
assert(max(res(14).x(:)-data{14}(:)) < 1e-3) ; % output relu+conv

% Block 6
assert(max(res(16).x(:)-data{15}(:)) < 1e-3) ; % output vec+fully

% Block 7
assert(max(res(18).x(:)-data{16}(:)) < 1e-3) ; % output relu+fully

% Block 8
assert(max(res(20).x(:)-data{17}(:)) < 1e-3) ; % output relu+fully

% Block 9
assert(max(res(22).x(:)-data{18}(:)) < 1e-3) ; % output relu+fully

% Block 10
assert(max(res(24).x(:)-data{19}(:)) < 1e-3) ; % output relu+softmax




