function [net, info] = example_rnn(varargin)
% EXAMPLE_RNN  Demonstrates MatConNet RNN trained on Shakespeare text.

run(fullfile(fileparts(mfilename('fullpath')),...
  '..', '..', 'matlab', 'vl_setupnn.m')) ;

opts.expDir = fullfile(vl_rootnn, 'data', 'rnn') ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.dataDir = fullfile(vl_rootnn, 'data', 'text') ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');

opts.train.batchSize = 100 ;
opts.train.numEpochs = 20 ;
opts.train.continue = true ;
opts.train.gpus = [] ;
opts.train.learningRate = 0.001 ;
opts.train.expDir = opts.expDir ;
opts.train.numSubBatches = 1 ;
opts = vl_argparse(opts, varargin) ;

% --------------------------------------------------------------------
%                                                         Prepare data
% --------------------------------------------------------------------

if exist(opts.imdbPath, 'file')
  imdb = load(opts.imdbPath) ;
else
  imdb = getImdb(opts) ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end


rng('default') ;
rng(0) ;

vocabSize = numel(imdb.vocabulary) ;  % number of characters in vocabulary
d = 100 ;  % number of hidden units

text = Input() ;  % size = [vocabulary size, batch size, phrase size]

Wi = Param('value', xavier(d, vocabSize)) ;  % input weights
Wh = Param('value', xavier(d, d)) ;          % hidden weights
bh = Param('value', zeros(1, d, 'single')) ; % hidden biases
Wo = Param('value', xavier(vocabSize, d)) ;  % output weights
bo = Param('value', zeros(1, d, 'single')) ; % output biases

h = zeros(d, 1, 'single') ;  % initial state
i = text(:,:,1) ;  % initial input

loss = 0 ;
error = 0 ;

for k = 1 : imdb.phraseSize - 1
  % new hidden state
  h = vl_nnsigmoid(Wi * i + Wh * h + bh) ;
  
  % unnormalized log-likelihood of next characters
  o = Wo * h + bo ;
  
  % get next input and compute prediction loss
  i = text(:,:,k+1) ;
  loss = loss + vl_nnloss(o, i, 'loss', 'softmaxlog') ;
  
  % classification error
  error = error + vl_nnloss(o, i, 'loss', 'classerror') ;
end

Layer.autoNames() ;
net = Net(loss * error) ;  % using * to prevent merging sums; a better fix is to allow multiple Net outputs

opts.train.stats = {loss, error} ;  % plot these quantities

% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------

bopts.useGpu = numel(opts.train.gpus) > 0 ;

info = cnn_train_autonn(net, imdb, @(varargin) getBatch(bopts,varargin{:}), opts.train) ;

% --------------------------------------------------------------------
function x = xavier(varargin)
% --------------------------------------------------------------------
% xavier initialization for weights
x = randn(varargin{:}, 'single') / sqrt(2 * prod([varargin{1:end-1}])) ;

% --------------------------------------------------------------------
function inputs = getBatch(opts, imdb, batch)
% --------------------------------------------------------------------
% one-hot encoding of characters. returned data shape is:
% [encoding size, batch size, phrase size].

idx = permute(imdb.images.data(:,batch), [3 2 1]) ;
text = single(bsxfun(@eq, idx, (1:numel(imdb.vocabulary))')) ;

if opts.useGpu > 0
  text = gpuArray(text) ;
end
inputs = {'text', text} ;

% --------------------------------------------------------------------
function imdb = getImdb(opts)
% --------------------------------------------------------------------

% download the full works of Shakespeare from Project Gutenberg
if ~exist(fullfile(opts.dataDir, '100.txt'), 'file')
  url = 'http://www.gutenberg.org/files/100/100.zip' ;
  fprintf('downloading %s\n', url) ;
  unzip(url, opts.dataDir) ;
end

data = fileread(fullfile(opts.dataDir, '100.txt')) ;

% for now, we just consider each line a different sample, and pad with
% spaces to make them all conform to the same size.
data = strsplit(data, '\n') ;
data = char(data{:}).' ;  % concatenate lines and transpose (samples in columns)


% create vocabulary of characters, and convert data to indices
sz = size(data) ;
[vocabulary, ~, data] = unique(data) ;
data = uint8(reshape(data, sz)) ;

set = ones(sz(2), 1) ;
set(randperm(numel(set), ceil(numel(set) * 0.2))) = 2;  % random 20% as val set

imdb.images.data = data ;
imdb.images.set = set ;
imdb.vocabulary = vocabulary ;
imdb.phraseSize = sz(1) ;
imdb.meta.sets = {'train', 'val'} ;

