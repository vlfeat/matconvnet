function [net, info] = cnn_mnist_autonn(varargin)
% CNN_MNIST_AUTONN  Demonstrates MatConNet on MNIST using AutoNN

run(fullfile(fileparts(mfilename('fullpath')),...
  '..', '..', 'matlab', 'vl_setupnn.m')) ;

% opts.modelType = 'mlp' ;
opts.modelType = 'lenet' ;
[opts, varargin] = vl_argparse(opts, varargin) ;

sfx = opts.modelType ;
opts.expDir = fullfile(vl_rootnn, 'data', ['mnist-baseline' sfx]) ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.dataDir = fullfile(vl_rootnn, 'data', 'mnist') ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');

opts.train.batchSize = 100 ;
opts.train.numEpochs = 20 ;
opts.train.continue = true ;
opts.train.gpus = [] ;
opts.train.learningRate = 0.00001 ;  %0.001 ;
opts.train.expDir = opts.expDir ;
opts.train.numSubBatches = 1 ;
opts = vl_argparse(opts, varargin) ;

% --------------------------------------------------------------------
%                                                         Prepare data
% --------------------------------------------------------------------

if exist(opts.imdbPath, 'file')
  imdb = load(opts.imdbPath) ;
else
  imdb = getMnistImdb(opts) ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end


rng('default') ;
rng(0) ;

images = Input() ;
labels = Input() ;

switch opts.modelType
case 'mlp'  % multi-layer perceptron
  x = vl_nnconv(images, 'size', [28, 28, 1, 100]) ;
  x = vl_nnrelu(x) ;
  x = vl_nnconv(x, 'size', [1, 1, 100, 100]) ;
  x = vl_nnrelu(x) ;
  x = vl_nndropout(x, 'rate', 0.5) ;
  x = vl_nnconv(x, 'size', [1, 1, 100, 10]) ;
    
case 'lenet'  % LeNet
  x = vl_nnconv(images, 'size', [5, 5, 1, 20]) ;
  x = vl_nnpool(x, 2, 'stride', 2) ;
  x = vl_nnconv(x, 'size', [5, 5, 20, 50]) ;
  x = vl_nnpool(x, 2, 'stride', 2) ;
  x = vl_nnconv(x, 'size', [4, 4, 50, 500]) ;
  x = vl_nnrelu(x) ;
  x = vl_nnconv(x, 'size', [1, 1, 500, 10]) ;
  
end

objective = vl_nnloss(x, labels, 'loss', 'softmaxlog') ;
error = vl_nnloss(x, labels, 'loss', 'classerror') ;

Layer.autoNames() ;
net = Net(objective + error) ;

% rng(0) ;

% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------

bopts.useGpu = numel(opts.train.gpus) >  0 ;

info = cnn_train_autonn(net, imdb, @(varargin) getBatch(bopts,varargin{:}), ...
                     opts.train, 'val', find(imdb.images.set == 3)) ;

% --------------------------------------------------------------------
function inputs = getBatch(opts, imdb, batch)
% --------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ;
if opts.useGpu > 0
  images = gpuArray(images) ;
end
inputs = {'images', images, 'labels', imdb.images.labels(1,batch)} ;

% --------------------------------------------------------------------
function imdb = getMnistImdb(opts)
% --------------------------------------------------------------------
% Preapre the imdb structure, returns image data with mean image subtracted
files = {'train-images-idx3-ubyte', ...
         'train-labels-idx1-ubyte', ...
         't10k-images-idx3-ubyte', ...
         't10k-labels-idx1-ubyte'} ;

if ~exist(opts.dataDir, 'dir')
  mkdir(opts.dataDir) ;
end

for i=1:4
  if ~exist(fullfile(opts.dataDir, files{i}), 'file')
    url = sprintf('http://yann.lecun.com/exdb/mnist/%s.gz',files{i}) ;
    fprintf('downloading %s\n', url) ;
    gunzip(url, opts.dataDir) ;
  end
end

f=fopen(fullfile(opts.dataDir, 'train-images-idx3-ubyte'),'r') ;
x1=fread(f,inf,'uint8') ;
fclose(f) ;
x1=permute(reshape(x1(17:end),28,28,60e3),[2 1 3]) ;

f=fopen(fullfile(opts.dataDir, 't10k-images-idx3-ubyte'),'r') ;
x2=fread(f,inf,'uint8') ;
fclose(f) ;
x2=permute(reshape(x2(17:end),28,28,10e3),[2 1 3]) ;

f=fopen(fullfile(opts.dataDir, 'train-labels-idx1-ubyte'),'r') ;
y1=fread(f,inf,'uint8') ;
fclose(f) ;
y1=double(y1(9:end)')+1 ;

f=fopen(fullfile(opts.dataDir, 't10k-labels-idx1-ubyte'),'r') ;
y2=fread(f,inf,'uint8') ;
fclose(f) ;
y2=double(y2(9:end)')+1 ;

set = [ones(1,numel(y1)) 3*ones(1,numel(y2))] ;
data = single(reshape(cat(3, x1, x2),28,28,1,[])) ;
dataMean = mean(data(:,:,:,set == 1), 4) ;
data = bsxfun(@minus, data, dataMean) ;

imdb.images.data = data ;
imdb.images.data_mean = dataMean ;
imdb.images.labels = cat(2, y1, y2) ;
imdb.images.set = set ;
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.classes = arrayfun(@(x)sprintf('%d',x),0:9,'uniformoutput',false) ;

