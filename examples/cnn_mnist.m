function [net, info] = cnn_mnist(varargin)
% CNN_MNIST  Demonstrated MatConNet on MNIST

run(fullfile(fileparts(mfilename('fullpath')),'..','matlab','vl_setupnn.m')) ;

opts.dataDir = fullfile('data','mnist') ;
opts.expDir = fullfile('data','mnist-baseline') ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.train.batchSize = 100 ;
opts.train.numEpochs = 100 ;
opts.train.continue = true ;
opts.train.useGpu = false ;
opts.train.learningRate = 0.001 ;
opts.train.expDir = opts.expDir ;
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

% Define a network similar to LeNet
f=1/100 ;
net.layers = {} ;
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', f*randn(5,5,1,20, 'single'), ...
                           'biases', zeros(1, 20, 'single'), ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', f*randn(5,5,20,50, 'single'),...
                           'biases', zeros(1,50,'single'), ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', f*randn(4,4,50,500, 'single'),...
                           'biases', zeros(1,500,'single'), ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'relu') ;
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', f*randn(1,1,500,10, 'single'),...
                           'biases', zeros(1,10,'single'), ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'softmaxloss') ;

% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------

% Take the mean out and make GPU if needed
imdb.images.data = bsxfun(@minus, imdb.images.data, mean(imdb.images.data,4)) ;
if opts.train.useGpu
  imdb.images.data = gpuArray(imdb.images.data) ;
end

[net, info] = cnn_train(net, imdb, @getBatch, ...
    opts.train, ...
    'val', find(imdb.images.set == 3)) ;

% --------------------------------------------------------------------
function [im, labels] = getBatch(imdb, batch)
% --------------------------------------------------------------------
im = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;

% --------------------------------------------------------------------
function imdb = getMnistImdb(opts)
% --------------------------------------------------------------------
files = {'train-images-idx3-ubyte', ...
         'train-labels-idx1-ubyte', ...
         't10k-images-idx3-ubyte', ...
         't10k-labels-idx1-ubyte'} ;

mkdir(opts.dataDir) ;
for i=1:4
  if ~exist(fullfile(opts.dataDir, files{i}), 'file')
    url = sprintf('http://yann.lecun.com/exdb/mnist/%s.gz',files{i}) ;
    fprintf('downloading %s\n', url) ;
    gunzip(url, opts.dataDir) ;
  end
end

f=fopen(fullfile(opts.dataDir, 'train-images-idx3-ubyte'),'r') ;
x1=fread(f,inf,'uint8');
fclose(f) ;
x1=permute(reshape(x1(17:end),28,28,60e3),[2 1 3]) ;

f=fopen(fullfile(opts.dataDir, 't10k-images-idx3-ubyte'),'r') ;
x2=fread(f,inf,'uint8');
fclose(f) ;
x2=permute(reshape(x2(17:end),28,28,10e3),[2 1 3]) ;

f=fopen(fullfile(opts.dataDir, 'train-labels-idx1-ubyte'),'r') ;
y1=fread(f,inf,'uint8');
fclose(f) ;
y1=double(y1(9:end)')+1 ;

f=fopen(fullfile(opts.dataDir, 't10k-labels-idx1-ubyte'),'r') ;
y2=fread(f,inf,'uint8');
fclose(f) ;
y2=double(y2(9:end)')+1 ;

imdb.images.data = single(reshape(cat(3, x1, x2),28,28,1,[])) ;
imdb.images.labels = cat(2, y1, y2) ;
imdb.images.set = [ones(1,numel(y1)) 3*ones(1,numel(y2))] ;
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.classes = arrayfun(@(x)sprintf('%d',x),0:9,'uniformoutput',false) ;
