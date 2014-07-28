function cnn_imagenet_evaluate()
% CNN_IMAGENET   Demonstrates MatConvNet on ImageNet

opts.dataDir = 'data/imagenet12' ;
opts.expDir = 'data/imagenet12-ref-models-eval' ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.lite = false ;
opts.train.batchSize = 256 ;
opts.train.numEpochs = 1 ;
opts.train.useGpu = false ;
opts.train.expDir = opts.expDir ;

run(fullfile(fileparts(mfilename('fullpath')), '../matlab/vl_setupnn.m')) ;

models = {...
  'data/caffe-ref-net.mat', ...
  'data/models/imagenet-caffe-ref.mat', ...
  'data/models/imagenet-vgg-f.mat', ...
  'data/models/imagenet-vgg-m.mat', ...
  'data/models/imagenet-vgg-s.mat'} ;

% -------------------------------------------------------------------------
%                                                   Database initialization
% -------------------------------------------------------------------------

if exist(opts.imdbPath)
  imdb = load(opts.imdbPath) ;
else
  imdb = cnn_imagenet_setup_data('dataDir', opts.dataDir, 'lite', opts.lite) ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end

% patch
%imdb.images.name = strrep(imdb.images.name, '.JPEG', '.jpg') ;

% -------------------------------------------------------------------------
%                                                    Network initialization
% -------------------------------------------------------------------------

net = load(models{1}) ;
sz=net.normalization.imageSize(1:2);
net.normalization.averageImage = ...
    imresize(net.normalization.averageImage, ...
             sz(:)');
net.layers{end}.type = 'softmaxloss' ; % softmax -> softmaxloss

% -------------------------------------------------------------------------
%                                               Stochastic gradient descent
% -------------------------------------------------------------------------

fn = getBatchWrapper(...
  net.normalization.averageImage, ...
  net.normalization.imageSize) ;

[net,info] = cnn_train(net, imdb, fn, opts.train, ...
  'conserveMemory', true, ...
  'train', NaN, ...
  'val', find(imdb.images.set==2)) ;

% -------------------------------------------------------------------------
function fn = getBatchWrapper(averageImage, size)
% -------------------------------------------------------------------------
fn = @(imdb,batch) cnn_imagenet_get_batch(imdb,batch,...
  'average',averageImage,...
  'size', size, ...
  'border', [0 0]) ;


