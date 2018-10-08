function [net, opts, imdb, info] = cnn_mnist_autoencoder
%CNN_MNIST_AUTOENCODER Summary of this function goes here
%   Detailed explanation goes here

net  = getMnistAutoencoderNet;
opts = getMnistAutoencoderOpts;

if exist(opts.imdbPath, 'file')
    
    load(opts.imdbPath);
    
else
    
    imdb = getMnistAutoencoderImdb(opts);
    
    if ~exist(opts.expDir, 'dir')
        
        mkdir(opts.expDir);
        
    end
    
    save(opts.imdbPath, 'imdb');
    
end

% [net, info] = cnn_train(net, imdb, @(imdb, batch) getMnistAutoencoderBatch(imdb, batch), opts);
[net, info] = cnn_train_adagrad(net, imdb, @(imdb, batch) getMnistAutoencoderBatch(imdb, batch), opts);

net.layers{end} = struct('name', 'data_hat_sigmoid', ...
                         'type', 'sigmoid'         );

net.layers{end + 1} = struct('type', 'euclideanloss');

end

% -------------------------------------------------------------------------
function net = getMnistAutoencoderNet
% -------------------------------------------------------------------------

% Layer 1

net.layers{1} = struct('biases'             , zeros(1, 1000, 'single')             , ...
                       'biasesLearningRate' , 1                                    , ...
                       'biasesWeightDecay'  , 0                                    , ...
                       'filters'            , sparse_initialization([1 1 784 1000]), ...
                       'filtersLearningRate', 1                                    , ...
                       'filtersWeightDecay' , 1                                    , ...
                       'name'               , 'encoder_1'                          , ...
                       'pad'                , [0 0 0 0]                            , ...
                       'stride'             , [1 1]                                , ...
                       'type'               , 'conv'                               );

net.layers{2} = struct('name', 'encoder_1_sigmoid', ...
                       'type', 'sigmoid'          );

% Layer 2

net.layers{3} = struct('biases'             , zeros(1, 500, 'single')              , ...
                       'biasesLearningRate' , 1                                    , ...
                       'biasesWeightDecay'  , 0                                    , ...
                       'filters'            , sparse_initialization([1 1 1000 500]), ...
                       'filtersLearningRate', 1                                    , ...
                       'filtersWeightDecay' , 1                                    , ...
                       'name'               , 'encoder_2'                          , ...
                       'pad'                , [0 0 0 0]                            , ...
                       'stride'             , [1 1]                                , ...
                       'type'               , 'conv'                               );

net.layers{4} = struct('name', 'encoder_2_sigmoid', ...
                       'type', 'sigmoid'          );

% Layer 3

net.layers{5} = struct('biases'             , zeros(1, 250, 'single')             , ...
                       'biasesLearningRate' , 1                                   , ...
                       'biasesWeightDecay'  , 0                                   , ...
                       'filters'            , sparse_initialization([1 1 500 250]), ...
                       'filtersLearningRate', 1                                   , ...
                       'filtersWeightDecay' , 1                                   , ...
                       'name'               , 'encoder_3'                         , ...
                       'pad'                , [0 0 0 0]                           , ...
                       'stride'             , [1 1]                               , ...
                       'type'               , 'conv'                              );

net.layers{6} = struct('name', 'encoder_3_sigmoid', ...
                       'type', 'sigmoid'          );

% Layer 4

net.layers{5} = struct('biases'             , zeros(1, 30, 'single')             , ...
                       'biasesLearningRate' , 1                                  , ...
                       'biasesWeightDecay'  , 0                                  , ...
                       'filters'            , sparse_initialization([1 1 250 30]), ...
                       'filtersLearningRate', 1                                  , ...
                       'filtersWeightDecay' , 1                                  , ...
                       'name'               , 'code'                             , ...
                       'pad'                , [0 0 0 0]                          , ...
                       'stride'             , [1 1]                              , ...
                       'type'               , 'conv'                             );

% Layer 5

net.layers{6} = struct('biases'             , zeros(1, 250, 'single')            , ...
                       'biasesLearningRate' , 1                                  , ...
                       'biasesWeightDecay'  , 0                                  , ...
                       'filters'            , sparse_initialization([1 1 30 250]), ...
                       'filtersLearningRate', 1                                  , ...
                       'filtersWeightDecay' , 1                                  , ...
                       'name'               , 'decoder_3'                        , ...
                       'pad'                , [0 0 0 0]                          , ...
                       'stride'             , [1 1]                              , ...
                       'type'               , 'conv'                             );

net.layers{7} = struct('name', 'decoder_3_sigmoid', ...
                       'type', 'sigmoid'          );

% Layer 6

net.layers{8} = struct('biases'             , zeros(1, 500, 'single')             , ...
                       'biasesLearningRate' , 1                                   , ...
                       'biasesWeightDecay'  , 0                                   , ...
                       'filters'            , sparse_initialization([1 1 250 500]), ...
                       'filtersLearningRate', 1                                   , ...
                       'filtersWeightDecay' , 1                                   , ...
                       'name'               , 'decoder_2'                         , ...
                       'pad'                , [0 0 0 0]                           , ...
                       'stride'             , [1 1]                               , ...
                       'type'               , 'conv'                              );

net.layers{9} = struct('name', 'decoder_2_sigmoid', ...
                       'type', 'sigmoid'          );

% Layer 7

net.layers{10} = struct('biases'             , zeros(1, 1000, 'single')             , ...
                        'biasesLearningRate' , 1                                    , ...
                        'biasesWeightDecay'  , 0                                    , ...
                        'filters'            , sparse_initialization([1 1 500 1000]), ...
                        'filtersLearningRate', 1                                    , ...
                        'filtersWeightDecay' , 1                                    , ...
                        'name'               , 'decoder_1'                          , ...
                        'pad'                , [0 0 0 0]                            , ...
                        'stride'             , [1 1]                                , ...
                        'type'               , 'conv'                               );

net.layers{11} = struct('name', 'decoder_1_sigmoid', ...
                        'type', 'sigmoid'          );

% Layer 8

net.layers{12} = struct('biases'             , zeros(1, 784, 'single')              , ...
                        'biasesLearningRate' , 1                                    , ...
                        'biasesWeightDecay'  , 0                                    , ...
                        'filters'            , sparse_initialization([1 1 1000 784]), ...
                        'filtersLearningRate', 1                                    , ...
                        'filtersWeightDecay' , 1                                    , ...
                        'name'               , 'data_hat'                           , ...
                        'pad'                , [0 0 0 0]                            , ...
                        'stride'             , [1 1]                                , ...
                        'type'               , 'conv'                               );

net.layers{13} = struct('type', 'sigmoidcrossentropyloss');

vl_simplenn_display(net);

end

% -------------------------------------------------------------------------
function filters = sparse_initialization(d)
% -------------------------------------------------------------------------

filters = zeros(d, 'single');

for index = 1 : d(4)
    
    p = randperm(d(3), 15);
    
    filters(1, 1, p, index) = randn(1, 1, 15, 1);
    
end

end

% -------------------------------------------------------------------------
function opts = getMnistAutoencoderOpts
% -------------------------------------------------------------------------

opts.batchSize       = 100;
opts.conserveMemory  = false;
opts.continue        = false;
opts.dataDir         = fullfile('data','mnist');
opts.display         = 10;
opts.delta           = 1e-8;
opts.errorType       = 'euclideanloss';
opts.expDir          = fullfile('data','mnistAutoencoder');
opts.imdbPath        = fullfile(opts.expDir, 'imdb.mat');
% opts.learningRate    = 1e-4;
opts.learningRate    = 1e-2;
% opts.momentum        = 0.9;
% opts.numEpochs       = 6667; % 6667 epochs is ~4000000 iterations.
opts.numEpochs       = 108; % 108 epochs is ~65000 iterations.
opts.plotDiagnostics = false;
opts.prefetch        = false;
opts.snapshot        = 10;
opts.sync            = true;
opts.test_interval   = 10;
opts.train           = [];
opts.useGpu          = true;
opts.val             = [];
opts.weightDecay     = 5e-4;

end

% -------------------------------------------------------------------------
function imdb = getMnistAutoencoderImdb(opts)
% -------------------------------------------------------------------------
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

set = [ones(1,numel(y1)) 2*ones(1,numel(y2))];
% data = single(reshape(cat(3, x1, x2),28,28,1,[]));
% dataMean = mean(data(:,:,:,set == 1), 4);
% data = bsxfun(@minus, data, dataMean) ;
data = single(reshape(cat(3, x1, x2), 1, 1, 784, []));
data = data - min(data(:)); data = data / max(data(:));

imdb.images.data = data ;
% imdb.images.data_mean = dataMean;
imdb.images.labels = cat(2, y1, y2) ;
imdb.images.set = set ;
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.classes = arrayfun(@(x)sprintf('%d',x),0:9,'uniformoutput',false) ;

end

% -------------------------------------------------------------------------
function [im, labels] = getMnistAutoencoderBatch(imdb, batch)
% -------------------------------------------------------------------------

im     = imdb.images.data(:, :, :, batch);
labels = im;

end

