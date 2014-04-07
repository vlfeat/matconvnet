function test_tinynet_caffe()

if ~exist('net')
  net = load(fullfile(vlg_root, 'data', 'tinynet_caffe.mat')) ;
  load(fullfile(vlg_root, 'data', 'tinynet_caffe_data.mat')) ;
end

for gpu = [false true]
  dzdy = zeros(1000,1,'single') ;
  dzdy(286) = 1 ;
  if gpu
    net_ = tinynet_move(net, 'gpu') ;
    im_ = gpuArray(im) ;
    dzdy_ = gpuArray(dzdy) ;
    res = tinynet(net_, im_, dzdy_) ;
  else
    res = tinynet(net, im, dzdy) ;
  end
  pairs = [1  1  % 1:  data
           3  2  % 2:  conv + relu
           4  3  % 3:  pool
           5  4  % 4:  normalize
           7  6  % 5:  pad + conv + relu
           8  7  % 6:  pool
           9  8  % 7:  normalize
           11 10 % 8:  pad + conv + relu
           13 12 % 9:  pad + conv + relu
           15 14 % 10: pad + conv + relu
           16 15 % 11: pool
           19 16 % 12: vec+fully+relu
           21 17 % 13: fully+relu
           22 18 % 14: fully
           23 19 % 15: softmax
          ] ;

  % Values x
  for i=1:size(pairs,1)
    testsim(res(pairs(i,1)).x(:), data{pairs(i,2)}(:)) ;
  end

  % Derivatives dzdx
  for i=1:size(pairs,1)
    testsim(res(pairs(i,1)).dzdx(:), diff{pairs(i,2)}(:)) ;
  end

  % Derivatives dzdw
  for i = 1:2
    testsim(res(1).dzdw{i}(:),pdiff{2}{i}(:)) ;
    testsim(res(5).dzdw{i}(:),pdiff{6}{i}(:)) ;
    testsim(res(9).dzdw{i}(:),pdiff{10}{i}(:)) ;
    testsim(res(11).dzdw{i}(:),pdiff{12}{i}(:)) ;
    testsim(res(13).dzdw{i}(:),pdiff{14}{i}(:)) ;
    testsim(res(17).dzdw{i}(:),pdiff{16}{i}(:)) ;
    testsim(res(19).dzdw{i}(:),pdiff{17}{i}(:)) ;
    testsim(res(21).dzdw{i}(:),pdiff{18}{i}(:)) ;
  end
end
