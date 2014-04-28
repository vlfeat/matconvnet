function vl_test_simplenn_caffe()

if ~exist('net')
  net = load(fullfile(vl_rootnn, 'data', 'tinynet_caffe.mat')) ;
  load(fullfile(vl_rootnn, 'data', 'tinynet_caffe_data.mat')) ;
end

%vl_simplenn_display(net) ;
%return

for gpu = [false true]
  dzdy = zeros(1,1,1000,'single') ;
  dzdy(286) = 1 ;
  if gpu
    net_ = vl_simplenn_move(net, 'gpu') ;
    im_ = gpuArray(im) ;
    dzdy_ = gpuArray(dzdy) ;
    res = vl_simplenn(net_, im_, dzdy_) ;
  else
    res = vl_simplenn(net, im, dzdy) ;
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
           18 16 % 12: vec + fully + relu
           20 17 % 13: fully+relu
           21 18 % 14: fully
           22 19 % 15: softmax
          ] ;

  % Values x
  for i=1:size(pairs,1)
    vl_testsim(res(pairs(i,1)).x(:), data{pairs(i,2)}(:)) ;
  end

  % Derivatives dzdx
  for i=1:size(pairs,1)
    vl_testsim(res(pairs(i,1)).dzdx(:), diff{pairs(i,2)}(:)) ;
  end

  % Derivatives dzdw
  for i = 1:2
    vl_testsim(res(1).dzdw{i}(:),pdiff{2}{i}(:)) ;
    vl_testsim(res(5).dzdw{i}(:),pdiff{6}{i}(:)) ;
    vl_testsim(res(9).dzdw{i}(:),pdiff{10}{i}(:)) ;
    vl_testsim(res(11).dzdw{i}(:),pdiff{12}{i}(:)) ;
    vl_testsim(res(13).dzdw{i}(:),pdiff{14}{i}(:)) ;
    vl_testsim(res(16).dzdw{i}(:),pdiff{16}{i}(:)) ;
    vl_testsim(res(18).dzdw{i}(:),pdiff{17}{i}(:)) ;
    vl_testsim(res(20).dzdw{i}(:),pdiff{18}{i}(:)) ;
  end
end
