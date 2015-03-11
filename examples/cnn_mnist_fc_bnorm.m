function [net, info] = cnn_mnist_fc_bnorm(varargin)
% CNN_MNIST  Demonstrated MatConNet on MNIST

opts.dataDir = fullfile('data','mnist') ;
opts.expDir = fullfile('data','mnist-3xfc-bnorm') ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.nonlin = 'relu';
opts.useBnorm = true;
opts.train.batchSize = 60 ;
opts.train.numEpochs = 50 ;
opts.train.continue = true ;
opts.train.useGpu = true ;
opts.train.learningRate = 0.001 ; % Sigmoid learning rate - 1
opts.train.epochSize = 1000;
opts.train.plot_res = true;
opts.train.plotDiagnostics = false;
opts = vl_argparse(opts, varargin) ;

opts.train.expDir = opts.expDir ;

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
                           'filters', f*randn(28,28,1,100, 'single'), ...
                           'biases', [], ...
                           'stride', 1, ...
                           'pad', 0) ;
if opts.useBnorm, net.layers{end+1} = add_bnorm(net.layers{end}); end
net.layers{end+1} = struct('type', opts.nonlin) ;

net.layers{end+1} = struct('type', 'conv', ...
                           'filters', f*randn(1,1,100,100, 'single'),...
                           'biases', [], ...
                           'stride', 1, ...
                           'pad', 0) ;
if opts.useBnorm, net.layers{end+1} = add_bnorm(net.layers{end}); end
net.layers{end+1} = struct('type', opts.nonlin) ;

net.layers{end+1} = struct('type', 'conv', ...
                           'filters', f*randn(1,1,100,100, 'single'),...
                           'biases', [], ...
                           'stride', 1, ...
                           'pad', 0) ;
if opts.useBnorm, net.layers{end+1} = add_bnorm(net.layers{end}); end
net.layers{end+1} = struct('type', opts.nonlin) ;

net.layers{end+1} = struct('type', 'conv', ...
                           'filters', f*randn(1,1,100,10, 'single'),...
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

[net, info] = cnn_train(net, imdb, @getBatch, opts.train, ...
  'val', find(imdb.images.set == 3)) ;

% --------------------------------------------------------------------
function layer = add_bnorm(prev_layer)
% --------------------------------------------------------------------
assert(isfield(prev_layer, 'filters'));
ndim = size(prev_layer.filters, 4);
layer = struct('type', 'bnorm', ...
               'filters', ones(ndim, 1, 'single'), ...
               'biases', zeros(ndim, 1, 'single'), ...
               'biasesLearningRate', 1, ...
               'filtersLearningRate', 1, ...
               'filtersWeightDecay', 0, ...
               'biasesWeightDecay', 0) ;
  
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
