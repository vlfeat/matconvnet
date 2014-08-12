function cnn_imagenet_minimal()
% CNN_IMAGENET_MINIMAL   Minimalistic demonstration of how to run a CNN model

% setup toolbox
run(fullfile(fileparts(mfilename('fullpath')), '../matlab/vl_setupnn.m')) ;

% load a CNN
net = load('data/models/imagenet-caffe-ref.mat') ;

% load an image
im = imread('http://www.vlfeat.org/sandbox-matconvnet/caffe-cat.jpg') ;

% preprocess the image (see also CNN_IMAGENET_GET_BATCH)
im_ = single(im) ; % note: 255 range
im_ = imresize(im_, ...
  net.normalization.imageSize(1:2), ...
  net.normalization.interpolation) ;
im_ = im_ - net.normalization.averageImage ;

% evaluate the CNN
res = vl_simplenn(net, im_) ;

% show result
scores = squeeze(gather(res(end).x)) ;
[bestScore, best] = max(scores) ;
figure(1) ; clf ;
imagesc(im) ; title(sprintf('%s (%d), score %.3f',...
  net.classes.description{best}, best, bestScore)) ;
axis off
