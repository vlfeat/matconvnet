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
opts.clipGrad = 10 ;

opts.train.batchSize = 200 ;
opts.train.numEpochs = 20 ;
opts.train.continue = true ;
opts.train.gpus = [] ;
opts.train.learningRate = 0.01 ;
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

numChars = numel(imdb.vocabulary) ;  % number of characters in vocabulary
d = opts.numUnits ;  % number of hidden units
T = imdb.phraseLength ;

text = Input() ;  % size = [numChars, batchSize, T]


switch opts.model
case 'lstm'
  % initialize the shared parameters for an LSTM with d units
  [W, b] = vl_nnlstm_params(d, numChars) ;
  
  % initial state
  h = cell(T, 1);
  c = cell(T, 1);
  h{1} = Layer.zeros(d, size(text,2), 'single');
  c{1} = Layer.zeros(d, size(text,2), 'single');

  % compute LSTM hidden states for all time steps
  for t = 1 : T - 1
    [h{t+1}, c{t+1}] = vl_nnlstm(text(:,:,t), h{t}, c{t}, W, b, 'clipGrad', opts.clipGrad) ;
  end
  

case 'rnn'
  Wi = Param('value', 0.1 * randn(d, numChars, 'single')) ;  % input weights
  Wh = Param('value', 0.1 * randn(d, d, 'single')) ;         % hidden weights
  bh = Param('value', zeros(d, 1, 'single')) ;               % hidden biases
  
  % initial state
  h = cell(T, 1);
  h{1} = Layer.zeros(d, size(text,2), 'single');
  
  % compute RNN hidden states for all time steps
  for t = 1 : T - 1
    h{t+1} = vl_nnsigmoid(Wi * text(:,:,t) + Wh * h{t} + bh) ;
  end

otherwise
  error('Unknown model.') ;
end


% concatenate hidden states along 3rd dimension, ignoring initial state.
% H will have size [d, batchSize, T - 2]
H = cat(3, h{2:end}) ;

% final projection (note same projection is applied at all time steps)
prediction = vl_nnconv(permute(H, [3 4 1 2]), 'size', [1, 1, d, numChars]) ;


% the ground truth "next" char for each of the T-1 input chars
idx = Input() ;  % as indexes, not one-hot encodings
nextChar = permute(idx(1,:,2:T), [3 1 4 2]) ;

% compute loss and error
loss = vl_nnloss(prediction, nextChar, 'loss', 'softmaxlog') ;
err = vl_nnloss(prediction, nextChar, 'loss', 'classerror') ;


% assign names and compile network
Layer.workspaceNames() ;
net = Net(loss, err) ;

% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------

bopts.useGpu = numel(opts.train.gpus) > 0 ;

info = cnn_train_autonn(net, imdb, @(varargin) getBatch(bopts,varargin{:}), opts.train) ;

% --------------------------------------------------------------------
function inputs = getBatch(opts, imdb, batch)
% --------------------------------------------------------------------
% one-hot encoding of characters. returned data size is:
% [vocabSize, batchSize, phraseLength].

% permute to size [1, batchSize, phraseLength] tensor of character indexes
idx = permute(imdb.images.data(:,batch), [3 2 1]) ;

% now expand to one-hot encoding of those indexes, along the 1st dimension
text = single(bsxfun(@eq, idx, (1:numel(imdb.vocabulary))')) ;

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
imdb.phraseLength = sz(1) ;
imdb.meta.sets = {'train', 'val'} ;

