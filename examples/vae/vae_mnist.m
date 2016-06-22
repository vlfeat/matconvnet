function [net, info] = vae_mnist(varargin)
%CNN_MNIST  Demonstrates MatConvNet on MNIST

% run(fullfile(fileparts(mfilename('fullpath')),...
%   '..', '..', 'matlab', 'vl_setupnn.m')) ;

opts.networkType = 'dagnn' ;
opts.expDir = fullfile(vl_rootnn, 'data', 'mnist-vae') ;
opts.dataDir = fullfile(vl_rootnn, 'data', 'mnist') ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.learningRate = 0.001 ;
opts.weightDecay = 0.99 ;
opts.trainMethod='rmsprop';
opts.train = struct() ;
opts = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;

% --------------------------------------------------------------------
%                                                         Prepare data
% --------------------------------------------------------------------

net = vae_mnist_init;

if exist(opts.imdbPath, 'file')
  imdb = load(opts.imdbPath) ;
else
  imdb = getMnistImdb(opts) ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end

net.meta.classes.name = arrayfun(@(x)sprintf('%d',x),1:10,'UniformOutput',false) ;

% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------

[net, info] = cnn_train_dag(net, imdb, getBatch(opts), ...
  'expDir', opts.expDir, ...
  net.meta.trainOpts, ...
  opts.train, ...
  'val', find(imdb.images.set == 3)) ;

% --------------------------------------------------------------------
function fn = getBatch(opts)
% --------------------------------------------------------------------
    bopts = struct('numGpus', numel(opts.train.gpus)) ;
    fn = @(x,y) getDagNNBatch(bopts,x,y) ;

% --------------------------------------------------------------------
function inputs = getDagNNBatch(opts, imdb, batch)
% --------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ;
%labels = imdb.images.labels(1,batch) ;
if opts.numGpus > 0
  images = gpuArray(images) ;
end
inputs = {'input', images} ;

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
x1=im2single(fread(f,inf,'uint8'))/255;
fclose(f) ;
x1=reshape(x1(17:end),1,1,28*28,60e3) ;

f=fopen(fullfile(opts.dataDir, 't10k-images-idx3-ubyte'),'r') ;
x2=im2single(fread(f,inf,'uint8'))/255;
fclose(f) ;
x2=reshape(x2(17:end),1,1,28*28,10e3) ;

f=fopen(fullfile(opts.dataDir, 'train-labels-idx1-ubyte'),'r') ;
y1=fread(f,inf,'uint8');
fclose(f) ;
y1=double(y1(9:end)')+1 ;

f=fopen(fullfile(opts.dataDir, 't10k-labels-idx1-ubyte'),'r') ;
y2=fread(f,inf,'uint8');
fclose(f) ;
y2=double(y2(9:end)')+1 ;

set = [ones(1,numel(y1)) 3*ones(1,numel(y2))];
data = cat(4,x1,x2);
% dataMean = mean(data(:,:,:,set == 1), 4);
dataMean = 0;
% data = bsxfun(@minus, data, dataMean) ;

imdb.images.data = data ;
imdb.images.data_mean = dataMean;
imdb.images.labels = cat(2, y1, y2) ;
imdb.images.set = set ;
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.classes = arrayfun(@(x)sprintf('%d',x),0:9,'uniformoutput',false) ;
