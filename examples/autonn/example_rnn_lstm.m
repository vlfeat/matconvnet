function [net, info] = example_rnn_lstm(varargin)
% EXAMPLE_RNN_LSTM
% Demonstrates MatConNet RNN/LSTM trained on Shakespeare text.

run(fullfile(fileparts(mfilename('fullpath')),...
  '..', '..', 'matlab', 'vl_setupnn.m')) ;

opts.model = 'lstm' ;
opts.expDir = fullfile(vl_rootnn, 'data', 'rnn') ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.dataDir = fullfile(vl_rootnn, 'data', 'text') ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');

opts.numUnits = 100 ;

opts.train.batchSize = 100 ;
opts.train.numEpochs = 20 ;
opts.train.continue = true ;
opts.train.gpus = [] ;
if strcmp(opts.model, 'lstm')
  opts.train.learningRate = 0.1 ;
else
  opts.train.learningRate = 0.001 ;
end
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
d = opts.numUnits ;  % number of hidden units
T = imdb.phraseSize ;

text = Input() ;  % size = [phrase size, 1, vocabulary size, batch size]

idx = Input() ;
nextChar = idx(2:T,1,1,:) ;

switch opts.model
case 'lstm'
  % initialize the shared parameters for an LSTM with d units
  [W, b] = vl_nnlstm_params(d, vocabSize) ;
  
  % initial state
  h = cell(T, 1);
  c = cell(T, 1);
  h{1} = Layer.zeros(d, size(text,4), 'single');
  c{1} = Layer.zeros(d, size(text,4), 'single');

  % run LSTM over all time steps
  for t = 1 : T - 1
    [h{t+1}, c{t+1}] = vl_nnlstm(text(t,1,:,:), h{t}, c{t}, W, b, 'debug',true);
  end

  % concatenate hidden states along 3rd dimension, ignoring initial state.
  % H will have size [d, batch size, T - 2]
  H = cat(3, h{2:end}) ;

  % final projection (note same projection is applied at all time steps)
  prediction = vl_nnconv(permute(H, [3 4 1 2]), 'size', [1, 1, d, vocabSize]) ;

  loss = vl_nnloss(prediction, nextChar, 'loss', 'softmaxlog') ;
  err = vl_nnloss(prediction, nextChar, 'loss', 'classerror') ;
  

case 'rnn'
  Wi = Param('value', xavier(d, vocabSize)) ;  % input weights
  Wh = Param('value', xavier(d, d)) ;          % hidden weights
  bh = Param('value', zeros(1, d, 'single')) ; % hidden biases
  Wo = Param('value', xavier(vocabSize, d)) ;  % output weights
  bo = Param('value', zeros(1, d, 'single')) ; % output biases

  h = zeros(d, 1, 'single') ;  % initial state
  i = text(:,:,1) ;  % initial input

  loss = 0 ;
  err = 0 ;

  for k = 1 : imdb.phraseSize - 1
    % new hidden state
    h = vl_nnsigmoid(Wi * i + Wh * h + bh) ;

    % unnormalized log-likelihood of next characters
    o = Wo * h + bo ;

    % get next input and compute prediction loss
    i = text(:,:,k+1) ;
    loss = loss + vl_nnloss(o, i, 'loss', 'softmaxlog') ;

    % classification error
    err = err + vl_nnloss(o, i, 'loss', 'classerror') ;
  end

otherwise
  error('Unknown model.') ;
end

Layer.workspaceNames() ;
net = Net(loss, err) ;

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
% one-hot encoding of characters. returned data size is:
% [phrase size, 1, vocabulary size, batch size].

% reshape to [phrase size, 1, 1, batch size] tensor of character indexes
idx = reshape(imdb.images.data(:,batch), imdb.phraseSize, 1, 1, []) ;

% now expand to one-hot encoding of those indexes, along third dimension
text = single(bsxfun(@eq, idx, reshape(1:numel(imdb.vocabulary), 1, 1, []))) ;

if opts.useGpu > 0
  text = gpuArray(text) ;
end
inputs = {'text', text, 'idx', single(idx)} ;

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

