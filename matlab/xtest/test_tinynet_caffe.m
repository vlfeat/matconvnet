if ~exist('net')
  net = load('tinynet_caffe.mat') ;
  load('tinynet_caffe_data.mat') ;
end

% forward pass
res = tinynet(net, im) ;

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
assert(max(res(23).x(:)-data{19}(:)) < 1e-3) ; % output softmax

% Backward pass
dzdy = zeros(1000,1,'single') ;
dzdy(286) = 1 ;
res = tinynet(net, im, dzdy) ;

% Derivatives
assert(max(res(1).dzdx(:)-diff{1}(:)) < 1e-3) ;
assert(max(res(22).dzdx(:)-diff{18}(:)) < 1e-3) ;
assert(max(res(23).dzdx(:)-diff{19}(:)) < 1e-3) ; % softmax

for i = 1:2
  assert(max(res(1).dzdw{i}(:)-pdiff{2}{i}(:)) < 1e-3) ;
  assert(max(res(5).dzdw{i}(:)-pdiff{6}{i}(:)) < 1e-3) ;
  assert(max(res(9).dzdw{i}(:)-pdiff{10}{i}(:)) < 1e-3) ;
  assert(max(res(11).dzdw{i}(:)-pdiff{12}{i}(:)) < 1e-3) ;
  assert(max(res(13).dzdw{i}(:)-pdiff{14}{i}(:)) < 1e-3) ;
end
for i = 1:2
  assert(max(res(17).dzdw{i}(:)-pdiff{16}{i}(:)) < 1e-3) ;
  assert(max(res(19).dzdw{i}(:)-pdiff{17}{i}(:)) < 1e-3) ;
  assert(max(res(21).dzdw{i}(:)-pdiff{18}{i}(:)) < 1e-3) ;
end





