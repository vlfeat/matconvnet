function [net,stats] = cnn_train_autonn(net, imdb, getBatch, varargin)
%CNN_TRAIN_AUTONN Demonstrates training a CNN using the autonn framework
%    CNN_TRAIN_AUTONN() is similar to CNN_TRAIN(), but works with a Net
%    object.

% Copyright (C) 2016 Joao F. Henriques.
% Copyright (C) 2014-15 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

% Todo: save momentum with checkpointing (a waste?)

opts.expDir = fullfile('data','exp') ;
opts.continue = true ;
opts.batchSize = 256 ;
opts.numSubBatches = 1 ;
opts.train = [] ;
opts.val = [] ;
opts.gpus = [] ;
opts.prefetch = false ;
opts.numEpochs = 300 ;
opts.learningRate = 0.001 ;
opts.weightDecay = 0.0005 ;
opts.momentum = 0.9 ;
opts.memoryMapFile = fullfile(tempdir, 'matconvnet.bin') ;
opts.profile = false ;

opts.derOutputs = 1 ;
opts.stats = 'losses' ;  %list of layers that are stats (loss, error), their names, or 'losses' for automatic
opts.plotStatistics = true ;
opts.plotDiagnostics = false ;
opts.postEpochFn = [] ;  % postEpochFn(opts,epoch,net,stats) called after each epoch; can return a new learning rate, 0 to stop, [] for no change
opts = vl_argparse(opts, varargin) ;

if ~exist(opts.expDir, 'dir'), mkdir(opts.expDir) ; end
if isempty(opts.train), opts.train = find(imdb.images.set==1) ; end
if isempty(opts.val), opts.val = find(imdb.images.set==2) ; end
if isnan(opts.train), opts.train = [] ; end

% -------------------------------------------------------------------------
%                                                            Initialization
% -------------------------------------------------------------------------

state.getBatch = getBatch ;
evaluateMode = isempty(opts.train) ;
if ~evaluateMode
  if isempty(opts.derOutputs)
    error('DEROUTPUTS must be specified when training.\n') ;
  end
end

if isequal(opts.stats, 'losses')
  % by default, compute stats for loss layers. store their indexes.
  opts.stats = find(cellfun(@(f) isequal(f, @vl_nnloss) || ...
    isequal(f, @vl_nnsoftmaxloss), {net.forward.func})) ;
else
  % store indexes of the given stats layers (either objects or by name)
  assert(iscell(opts.stats)) ;
  names = {net.forward.name} ;
  stats = zeros(numel(opts.stats), 1) ;
  for i = 1 : numel(opts.stats)
    s = opts.stats{i} ;
    if isa(s, 'Layer')
      s = s.name ;
    end
    stats(i) = find(strcmp(names, s), 1) ;
    assert(~isempty(stats(i)), sprintf('Layer ''%s'' not found.', stats(i))) ;
  end
  opts.stats = stats ;
end
stats = [] ;

% get fan-out of each parameter; this is needed for trainMethod = 'average'
varInfo = net.getVarsInfo() ;
p = strcmp({varInfo.type}, 'param') & ~[varInfo.isDer] ;
state.paramsFanout = ones(nnz(p), 1) ;
state.paramsFanout([varInfo(p).index]) = cellfun(@numel, {varInfo(p).fanout}) ;

% setup GPUs
numGpus = numel(opts.gpus) ;
if numGpus > 1
  if isempty(gcp('nocreate')),
    parpool('local',numGpus) ;
    spmd, gpuDevice(opts.gpus(labindex)), end
  end
  if exist(opts.memoryMapFile)
    delete(opts.memoryMapFile) ;
  end
elseif numGpus == 1
  gpuDevice(opts.gpus)
end

% -------------------------------------------------------------------------
%                                                        Train and validate
% -------------------------------------------------------------------------

modelPath = @(ep) fullfile(opts.expDir, sprintf('net-epoch-%d.mat', ep)) ;
modelFigPath = fullfile(opts.expDir, 'net-train.pdf') ;

start = opts.continue * findLastCheckpoint(opts.expDir) ;
if start >= 1
  fprintf('%s: resuming by loading epoch %d\n', mfilename, start) ;
  [net, stats] = loadState(modelPath(start)) ;
end

for epoch=start+1:opts.numEpochs

  % train one epoch
  state.epoch = epoch ;
  state.learningRate = opts.learningRate(min(epoch, numel(opts.learningRate))) ;
  state.train = opts.train(randperm(numel(opts.train))) ; % shuffle
  state.val = opts.val ;
  state.imdb = imdb ;

  if numGpus <= 1
    stats.train(epoch) = process_epoch(net, state, opts, 'train') ;
    stats.val(epoch) = process_epoch(net, state, opts, 'val') ;
  else
    savedNet = net.saveobj() ;
    spmd
      net_ = dagnn.DagNN.loadobj(savedNet) ;
      stats_.train = process_epoch(net_, state, opts, 'train') ;
      stats_.val = process_epoch(net_, state, opts, 'val') ;
      if labindex == 1, savedNet_ = net_.saveobj() ; end
    end
    net = dagnn.DagNN.loadobj(savedNet_{1}) ;
    stats__ = accumulateStats(stats_) ;
    stats.train(epoch) = stats__.train ;
    stats.val(epoch) = stats__.val ;
    clear net_ stats_ stats__ savedNet_ ;
  end

  if ~evaluateMode
    saveState(modelPath(epoch), net, stats) ;
  end

  if opts.plotStatistics
    figure(1) ; clf ;
    plots = setdiff(...
      cat(2,...
      fieldnames(stats.train)', ...
      fieldnames(stats.val)'), {'num', 'time'}) ;
    for p = plots
      p = char(p) ;
      values = zeros(0, epoch) ;
      leg = {} ;
      for f = {'train', 'val'}
        f = char(f) ;
        if isfield(stats.(f), p)
          tmp = [stats.(f).(p)] ;
          values(end+1,:) = tmp(1,:)' ;
          leg{end+1} = f ;
        end
      end
      subplot(1,numel(plots),find(strcmp(p,plots))) ;
      plot(1:epoch, values','o-') ;
      xlabel('epoch') ;
      title(p) ;
      legend(leg{:}) ;
      grid on ;
    end
    drawnow ;
    try print(1, modelFigPath, '-dpdf') ; catch, end
  end
  
  if ~isempty(opts.postEpochFn)
    if nargout(opts.postEpochFn) == 0
      opts.postEpochFn(opts, epoch, net, stats) ;
    else
      lr = opts.postEpochFn(opts, epoch, net, stats) ;
      if ~isempty(lr), opts.learningRate = lr; end
      if opts.learningRate == 0, break; end
    end
  end
end

% -------------------------------------------------------------------------
function stats = process_epoch(net, state, opts, mode)
% -------------------------------------------------------------------------

if strcmp(mode,'train')
  state.momentum = num2cell(zeros(1, numel(net.params))) ;
end

numGpus = numel(opts.gpus) ;
if numGpus >= 1
  net.move('gpu') ;
  if strcmp(mode,'train')
    state.momentum = cellfun(@gpuArray,state.momentum,'UniformOutput',false) ;
  end
end
if numGpus > 1
  mmap = map_gradients(opts.memoryMapFile, net, numGpus) ;
else
  mmap = [] ;
end

statsAccum = zeros(numel(opts.stats), 1) ;
statsNames = {net.forward(opts.stats).name} ;
statsVars = [net.forward(opts.stats).outputVar] ;

% assign names automatically if needed
for i = 1:numel(statsNames)
  if isempty(statsNames{i})
    statsNames{i} = sprintf('stat%i', i) ;
  end
end

subset = state.(mode) ;
start = tic ;
num = 0 ;
time = 0 ;

for t=1:opts.batchSize:numel(subset)
  batchSize = min(opts.batchSize, numel(subset) - t + 1) ;

  for s=1:opts.numSubBatches
    % get this image batch and prefetch the next
    batchStart = t + (labindex-1) + (s-1) * numlabs ;
    batchEnd = min(t+opts.batchSize-1, numel(subset)) ;
    batch = subset(batchStart : opts.numSubBatches * numlabs : batchEnd) ;
    num = num + numel(batch) ;
    if numel(batch) == 0, continue ; end

    inputs = state.getBatch(state.imdb, batch) ;
    net.setInputs(inputs{:}) ;

    if opts.prefetch
      if s == opts.numSubBatches
        batchStart = t + (labindex-1) + opts.batchSize ;
        batchEnd = min(t+2*opts.batchSize-1, numel(subset)) ;
      else
        batchStart = batchStart + numlabs ;
      end
      nextBatch = subset(batchStart : opts.numSubBatches * numlabs : batchEnd) ;
      state.getBatch(state.imdb, nextBatch) ;
    end

    if strcmp(mode, 'train')
      net.eval('normal', opts.derOutputs, s ~= 1) ;
    else
      net.eval('test') ;
    end
    
    % accumulate learning stats
    for k = 1:numel(opts.stats)
      statsAccum(k) = statsAccum(k) + gather(net.vars{statsVars(k)}) ;
    end
  end

  % accumulate gradient
  if strcmp(mode, 'train')
    if ~isempty(mmap)
      write_gradients(mmap, net) ;
      labBarrier() ;
    end
    state = accumulate_gradients(state, net, opts, batchSize, mmap) ;
  end

  % print learning statistics
  time = toc(start) ;

  fprintf('%s: epoch %02d: %3d/%3d: %.1f Hz', ...
    mode, ...
    state.epoch, ...
    fix((t-1)/opts.batchSize)+1, ceil(numel(subset)/opts.batchSize), ...
    num/time * max(numGpus, 1)) ;

  for i = 1:numel(opts.stats)
    fprintf(' %s:', statsNames{i}) ;
    fprintf(' %.3f', statsAccum(i) / num) ;
  end
  fprintf('\n') ;
  
  if opts.plotDiagnostics && mod(t-1, opts.batchSize * 5) == 0
    net.plotDiagnostics(200) ;
  end
end

% return structure with statistics
stats.time = time ;
stats.num = num ;
for s = 1:numel(opts.stats)
  stats.(statsNames{s}) = statsAccum(s) / num ;
end

net.move('cpu') ;

% -------------------------------------------------------------------------
function state = accumulate_gradients(state, net, opts, batchSize, mmap)
% -------------------------------------------------------------------------

% ensure supported training methods are ordered as expected
assert(isequal(Param.trainMethods, {'gradient', 'average', 'none'})) ;

paramVars = [net.params.var] ;
w = net.getValue(paramVars) ;
dw = net.getDer(paramVars) ;

for p=1:numel(paramVars)
  % bring in gradients from other GPUs if any
  if ~isempty(mmap)
    error('Not implemented.') ;
    numGpus = numel(mmap.Data) ;
    tmp = zeros(size(mmap.Data(labindex).(net.params(p).name)), 'single') ;
    for g = setdiff(1:numGpus, labindex)
      tmp = tmp + mmap.Data(g).(net.params(p).name) ;
    end
    net.vars{varIdx} = net.vars{varIdx} + tmp ;
  end

  switch net.params(p).trainMethod
    case 1  % gradient
      thisDecay = opts.weightDecay * net.params(p).weightDecay ;
      thisLR = state.learningRate * net.params(p).learningRate ;
      state.momentum{p} = opts.momentum * state.momentum{p} ...
        - thisDecay * w{p} ...
        - (1 / batchSize) * dw{p} ;
      w{p} = w{p} + thisLR * state.momentum{p} ;

    case 2  % average, mainly for batch normalization
      thisLR = net.params(p).learningRate ;
      w{p} = (1 - thisLR) * w{p} + (thisLR/opts.numSubBatches) / state.paramsFanout(p) * dw{p} ;

    case 3  % none
    otherwise
      error('Unknown training method %i for parameter ''%s''.', ...
        net.params(p).trainMethod, ...
        net.params(p).name) ;
  end
end

net.setValue(paramVars, w) ;

% -------------------------------------------------------------------------
function mmap = map_gradients(fname, net, numGpus)
% -------------------------------------------------------------------------
format = {} ;
for i=1:numel(net.params)
  format(end+1,1:3) = {'single', size(net.params(i).value), net.params(i).name} ;
end
format(end+1,1:3) = {'double', [3 1], 'errors'} ;
if ~exist(fname) && (labindex == 1)
  f = fopen(fname,'wb') ;
  for g=1:numGpus
    for i=1:size(format,1)
      fwrite(f,zeros(format{i,2},format{i,1}),format{i,1}) ;
    end
  end
  fclose(f) ;
end
labBarrier() ;
mmap = memmapfile(fname, 'Format', format, 'Repeat', numGpus, 'Writable', true) ;

% -------------------------------------------------------------------------
function write_gradients(mmap, net)
% -------------------------------------------------------------------------
for i=1:numel(net.params)
  mmap.Data(labindex).(net.params(i).name) = gather(net.params(i).der) ;
end

% -------------------------------------------------------------------------
function stats = accumulateStats(stats_)
% -------------------------------------------------------------------------

stats = struct() ;

for s = {'train', 'val'}
  s = char(s) ;
  total = 0 ;

  for g = 1:numel(stats_)
    stats__ = stats_{g} ;
    num__ = stats__.(s).num ;
    total = total + num__ ;

    for f = setdiff(fieldnames(stats__.(s))', 'num')
      f = char(f) ;

      if g == 1
        stats.(s).(f) = 0 ;
      end
      stats.(s).(f) = stats.(s).(f) + stats__.(s).(f) * num__ ;

      if g == numel(stats_)
        stats.(s).(f) = stats.(s).(f) / total ;
      end
    end
  end
  stats.(s).num = total ;
end

% -------------------------------------------------------------------------
function saveState(fileName, net, stats)
% -------------------------------------------------------------------------
net_ = net ;
net = net_.saveobj() ;
save(fileName, 'net', 'stats') ;

% -------------------------------------------------------------------------
function [net, stats] = loadState(fileName)
% -------------------------------------------------------------------------
load(fileName, 'net', 'stats') ;
net = Net(net) ;

% -------------------------------------------------------------------------
function epoch = findLastCheckpoint(modelDir)
% -------------------------------------------------------------------------
list = dir(fullfile(modelDir, 'net-epoch-*.mat')) ;
tokens = regexp({list.name}, 'net-epoch-([\d]+).mat', 'tokens') ;
epoch = cellfun(@(x) sscanf(x{1}{1}, '%d'), tokens) ;
epoch = max([epoch 0]) ;
