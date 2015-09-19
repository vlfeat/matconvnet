function cnn_imagenet_googlenet()
%CNN_IMAGENET_GOOGLENET  Deomonstrates running GoogLeNet

run matlab/vl_setupnn
modelPath = 'data/models/imagenet-googlenet-dag.mat' ;

if ~exist(modelPath)
  mkdir(fileparts(modelPath)) ;
  urlwrite(...
  'http://www.vlfeat.org/matconvnet/models/imagenet-googlenet-dag.mat', ...
    modelPath) ;
end

netData = load('data/models/imagenet-googlenet-dag.mat') ;
net = dagnn.DagNN.loadobj(netData) ;
clear netData ;

net.removeLayer(net.layers(end).name) ;
net.addLayer('softmax', dagnn.SoftMax(), ...
             net.layers(end).outputs, {'prediction'}, {}) ;
net.mode = 'test' ;

im = imread('peppers.png') ;
im_ = single(im) ; % note: 255 range
im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
im_ = im_ - net.meta.normalization.averageImage ;
net.eval({'data', im_}) ;

% show the classification result
scores = squeeze(gather(net.vars(end).value)) ;
[bestScore, best] = max(scores) ;
figure(1) ; clf ; imagesc(im) ;
title(sprintf('%s (%d), score %.3f',...
net.meta.classes.description{best}, best, bestScore)) ;
