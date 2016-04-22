function [net, info] = cnn_stn_cluttered_mnist(varargin)
%CNN_STN_CLUTTERED_MNIST : demonstrates training a spatial transformer
%                          network (STN) on cluttered MNIST dataset.

run(fullfile(fileparts(mfilename('fullpath')),...
  '..', '..', 'matlab', 'vl_setupnn.m')) ;
opts.expDir = fullfile(vl_rootnn, 'data', 'cluttered-mnist') ;
[opts, varargin] = vl_argparse(opts, varargin) ;
opts.dataDir = fullfile(vl_rootnn, 'data', 'cluttered-mnist') ;
opts.dataDbFname = '/Users/ankushgupta/cdt/research/text_spotting/text-rnn/matconvnet-rnn/mnistUtils/cluttered_mnist.mat' ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.train = struct() ;
opts.train.numEpochs = 30 ;
opts.train.gpus = [] ; % add a gpu index here
opts.train.learningRate = 0.001 ;
opts = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;

% --------------------------------------------------------------------
%                                                         Prepare data
% --------------------------------------------------------------------
net = cnn_stn_cluttered_mnist_init([60 60]) ; % initialize the network
if exist(opts.imdbPath, 'file')
  imdb = load(opts.imdbPath) ;
else
  imdb = getImdb(opts) ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end
net.meta.classes.name = arrayfun(@(x)sprintf('%d',x),1:10,'UniformOutput',false) ;

% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------
fbatch = @(i,b) getBatch(opts.train,i,b);
[net, info] = cnn_train_dag(net, imdb, fbatch, ...
                            'expDir', opts.expDir, ...
                            opts.train, ...
                            'val', find(imdb.images.set == 2)) ;

% --------------------------------------------------------------------
function inputs = getBatch(opts, imdb, batch)
% --------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;
if opts.gpus > 0
  images = gpuArray(images) ;
end
inputs = {'input', images, 'label', labels} ;

% --------------------------------------------------------------------
function imdb = getImdb(opts)
% --------------------------------------------------------------------
% Prepare the IMDB structure:
if ~exist(opts.dataDir, 'dir')
  mkdir(opts.dataDir) ;
end
dat = load(opts.dataDbFname);
set = [ones(1,numel(dat.y_tr)) 2*ones(1,numel(dat.y_vl)) 3*ones(1,numel(dat.y_ts))];
data = single(cat(4,dat.x_tr,dat.x_vl,dat.x_ts));
imdb.images.data = data ;
imdb.images.labels = single(cat(2, dat.y_tr,dat.y_vl,dat.y_ts)) ;
imdb.images.set = set ;
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.classes = arrayfun(@(x)sprintf('%d',x),0:9,'uniformoutput',false) ;

