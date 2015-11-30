function cnn_vgg_faces()
%CNN_VGG_FACES  Demonstrates how to use VGG-Face

run matlab/vl_setupnn
modelPath = 'data/models/vgg-face.mat' ;

if ~exist(modelPath)
  mkdir(fileparts(modelPath)) ;
  urlwrite(...
  'http://www.vlfeat.org/matconvnet/models/vgg-face.mat', ...
    modelPath) ;
end

net = load('data/models/vgg-face.mat') ;

im = imread('https://upload.wikimedia.org/wikipedia/commons/4/4a/Aamir_Khan_March_2015.jpg') ;
im = im(1:250,:,:) ; % crop
im_ = single(im) ; % note: 255 range
im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
im_ = bsxfun(@minus,im_,net.meta.normalization.averageImage) ;
res = vl_simplenn(net, im_) ;

% show the classification result
scores = squeeze(gather(res(end).x)) ;
[bestScore, best] = max(scores) ;
figure(1) ; clf ; imagesc(im) ; axis equal off ;
title(sprintf('%s (%d), score %.3f',...
              net.meta.classes.description{best}, best, bestScore), ...
      'Interpreter', 'none') ;
