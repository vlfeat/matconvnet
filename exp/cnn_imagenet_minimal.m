function cnn_imagenet_minimal()
% CNN_IMAGENET_MINIMAL

% setup toolbox
run(fullfile(fileparts(mfilename('fullpath')), '../matlab/vl_setupnn.m')) ;

% load reference model
net = load('data/caffe-ref-net.mat') ;

% load an image
im = imread('http://www.vlfeat.org/sandbox-matconvnet/caffe-cat.jpg') ;

% preprocess the image
im_ = single(im) ; % note: 255 range
im_ = imresize(im_, net.normalization.imageSize, 'bilinear') ;
im_ = im_ - net.normalization.averageImage ;

% evaluate the CNN
res = vl_simplenn(net, im_) ;

% show result
scores = squeeze(gather(res(end).x)) ;
[bestScore, best] = max(scores) ;
figure(1) ; clf ;
imagesc(im) ; title(sprintf('%s (%d), score %f',...
                            net.classes{best}, best, bestScore)) ;

