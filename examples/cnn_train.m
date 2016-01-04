function [net, info] = cnn_train(net, imdb, getBatch, varargin)
%CNN_TRAIN  An example implementation of SGD for training CNNs
%    CNN_TRAIN() is an example learner implementing stochastic
%    gradient descent with momentum to train a CNN. It can be used
%    with different datasets and tasks by providing a suitable
%    getBatch function.
%
%    The function automatically restarts after each training epoch by
%    checkpointing.
%
%    The function supports training on CPU or on one or more GPUs
%    (specify the list of GPU IDs in the `gpus` option). Multi-GPU
%    support is relatively primitive but sufficient to obtain a
%    noticable speedup.

% Copyright (C) 2014-15 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

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

opts.conserveMemory = true ;
opts.backPropDepth = +inf ;
opts.sync = false ;
opts.cudnn = true ;
opts.errorFunction = 'multiclass' ;
opts.errorLabels = {} ;
opts.plotDiagnostics = false ;
opts = vl_argparse(opts, varargin) ;

if ~exist(opts.expDir, 'dir'), mkdir(opts.expDir) ; end
if isempty(opts.train), opts.train = find(imdb.images.set==1) ; end
if isempty(opts.val), opts.val = find(imdb.images.set==2) ; end
if isnan(opts.train), opts.train = [] ; end
if isnan(opts.val), opts.val = [] ; end

% -------------------------------------------------------------------------
%                                                    Network initialization
% -------------------------------------------------------------------------

vl_simplenn_display(net, 'batchSize', opts.batchSize) ;

evaluateMode = isempty(opts.train) ;

if ~evaluateMode
  for i=1:numel(net.layers)
    if isfield(net.layers{i}, 'weights')
      J = numel(net.layers{i}.weights) ;
      for j=1:J
        net.layers{i}.momentum{j} = zeros(size(net.layers{i}.weights{j}), 'single') ;
      end
      if ~isfield(net.layers{i}, 'learningRate')
        net.layers{i}.learningRate = ones(1, J, 'single') ;
      end
      if ~isfield(net.layers{i}, 'weightDecay')
        net.layers{i}.weightDecay = ones(1, J, 'single') ;
      end
    end
  end
end

% setup GPUs
numGpus = numel(opts.gpus) ;
if numGpus > 1
  if isempty(gcp('nocreate')),
    parpool('local',numGpus) ;
    spmd, gpuDevice(opts.gpus(labindex)), end
  end
elseif numGpus == 1
  gpuDevice(opts.gpus)
end
if exist(opts.memoryMapFile), delete(opts.memoryMapFile) ; end

% setup error calculation function
if isstr(opts.errorFunction)
  switch opts.errorFunction
    case 'none'
      opts.errorFunction = @error_none ;
    case 'multiclass'
      opts.errorFunction = @error_multiclass ;
      if isempty(opts.errorLabels), opts.errorLabels = {'top1err', 'top5err'} ; end
    case 'binary'
      opts.errorFunction = @error_binary ;
      if isempty(opts.errorLabels), opts.errorLabels = {'binerr'} ; end
    otherwise
      error('Unknown error function ''%s''.', opts.errorFunction) ;
  end
end

% -------------------------------------------------------------------------
%                                                        Train and validate
% -------------------------------------------------------------------------

modelPath = @(ep) fullfile(opts.expDir, sprintf('net-epoch-%d.mat', ep));
modelFigPath = fullfile(opts.expDir, 'net-train.pdf') ;

start = opts.continue * findLastCheckpoint(opts.expDir) ;
if start >= 1
  fprintf('%s: resuming by loading epoch %d\n', mfilename, start) ;
  load(modelPath(start), 'net', 'info') ;
  net = vl_simplenn_tidy(net) ; % just in case MatConvNet was updated
end

for epoch=start+1:opts.numEpochs

  % train one epoch and validate
  learningRate = opts.learningRate(min(epoch, numel(opts.learningRate))) ;
  train = opts.train(randperm(numel(opts.train))) ; % shuffle
  val = opts.val ;

  if numGpus <= 1
    [net,stats.train,prof] = process_epoch(opts, getBatch, epoch, train, learningRate, imdb, net) ;
    [~,stats.val] = process_epoch(opts, getBatch, epoch, val, 0, imdb, net) ;
    if opts.profile
      profile('viewer') ;
      keyboard ;
    end
  else
    fprintf('%s: sending model to %d GPUs\n', mfilename, numGpus) ;
    spmd(numGpus)
      [net_, stats_train_,prof_] = process_epoch(opts, getBatch, epoch, train, learningRate, imdb, net) ;
      [~, stats_val_] = process_epoch(opts, getBatch, epoch, val, 0, imdb, net_) ;
    end
    net = net_{1} ;
    stats.train = sum([stats_train_{:}],2) ;
    stats.val = sum([stats_val_{:}],2) ;
    if opts.profile
      mpiprofile('viewer', [prof_{:,1}]) ;
      keyboard ;
    end
    clear net_ stats_train_ stats_val_ ;
  end

  % save
  if evaluateMode, sets = {'val'} ; else sets = {'train', 'val'} ; end
  for f = sets
    f = char(f) ;
    n = numel(eval(f)) ;
    info.(f).speed(epoch) = n / stats.(f)(1) * max(1, numGpus) ;
    info.(f).objective(epoch) = stats.(f)(2) / n ;
    info.(f).error(:,epoch) = stats.(f)(3:end) / n ;
  end
  if ~evaluateMode
    fprintf('%s: saving model for epoch %d\n', mfilename, epoch) ;
    tic ;
    save(modelPath(epoch), 'net', 'info') ;
    fprintf('%s: model saved in %.2g s\n', mfilename, toc) ;
  end

  figure(1) ; clf ;
  hasError = isa(opts.errorFunction, 'function_handle') ;
  subplot(1,1+hasError,1) ;
  if ~evaluateMode
    semilogy(1:epoch, info.train.objective, '.-', 'linewidth', 2) ;
    hold on ;
  end
  semilogy(1:epoch, info.val.objective, '.--') ;
  xlabel('training epoch') ; ylabel('energy') ;
  grid on ;
  h=legend(sets) ;
  set(h,'color','none');
  title('objective') ;
  if hasError
    subplot(1,2,2) ; leg = {} ;
    if ~evaluateMode
      plot(1:epoch, info.train.error', '.-', 'linewidth', 2) ;
      hold on ;
      leg = horzcat(leg, strcat('train ', opts.errorLabels)) ;
    end
    plot(1:epoch, info.val.error', '.--') ;
    leg = horzcat(leg, strcat('val ', opts.errorLabels)) ;
    set(legend(leg{:}),'color','none') ;
    grid on ;
    xlabel('training epoch') ; ylabel('error') ;
    title('error') ;
  end
  drawnow ;
  print(1, modelFigPath, '-dpdf') ;
end

% -------------------------------------------------------------------------
function err = error_multiclass(opts, labels, res)
% -------------------------------------------------------------------------
predictions = gather(res(end-1).x) ;
[~,predictions] = sort(predictions, 3, 'descend') ;

% be resilient to badly formatted labels
if numel(labels) == size(predictions, 4)
  labels = reshape(labels,1,1,1,[]) ;
end

% skip null labels
mass = single(labels(:,:,1,:) > 0) ;
if size(labels,3) == 2
  % if there is a second channel in labels, used it as weights
  mass = mass .* labels(:,:,2,:) ;
  labels(:,:,2,:) = [] ;
end

error = ~bsxfun(@eq, predictions, labels) ;
err(1,1) = sum(sum(sum(mass .* error(:,:,1,:)))) ;
err(2,1) = sum(sum(sum(mass .* min(error(:,:,1:5,:),[],3)))) ;

% -------------------------------------------------------------------------
function err = error_binary(opts, labels, res)
% -------------------------------------------------------------------------
predictions = gather(res(end-1).x) ;
error = bsxfun(@times, predictions, labels) < 0 ;
err = sum(error(:)) ;

% -------------------------------------------------------------------------
function err = error_none(opts, labels, res)
% -------------------------------------------------------------------------
err = zeros(0,1) ;

% -------------------------------------------------------------------------
function  [net_cpu,stats,prof] = process_epoch(opts, getBatch, epoch, subset, learningRate, imdb, net_cpu)
% -------------------------------------------------------------------------

% move CNN to GPU as needed
numGpus = numel(opts.gpus) ;
if numGpus >= 1
  net = vl_simplenn_move(net_cpu, 'gpu') ;
else
  net = net_cpu ;
  net_cpu = [] ;
end

% validation mode if learning rate is zero
training = learningRate > 0 ;
if training
  mode = 'train' ;
  evalMode = 'normal' ;
else
  mode = 'val' ;
  evalMode = 'test' ;
end

% profile
if opts.profile
  if numGpus <= 1
    prof = profile('info') ;
    profile clear ;
    profile on ;
  else
    prof = mpiprofile('info') ;
    mpiprofile reset ;
    mpiprofile on ;
  end
end

numGpus = numel(opts.gpus) ;
if numGpus >= 1
  one = gpuArray(single(1)) ;
else
  one = single(1) ;
end
res = [] ;
mmap = [] ;
stats = [] ;
start = tic ;

for t=1:opts.batchSize:numel(subset)
  fprintf('%s: epoch %02d: %3d/%3d: ', mode, epoch, ...
          fix(t/opts.batchSize)+1, ceil(numel(subset)/opts.batchSize)) ;
  batchSize = min(opts.batchSize, numel(subset) - t + 1) ;
  numDone = 0 ;
  error = [] ;
  for s=1:opts.numSubBatches
    % get this image batch and prefetch the next
    batchStart = t + (labindex-1) + (s-1) * numlabs ;
    batchEnd = min(t+opts.batchSize-1, numel(subset)) ;
    batch = subset(batchStart : opts.numSubBatches * numlabs : batchEnd) ;
    [im, labels] = getBatch(imdb, batch) ;

    if opts.prefetch
      if s==opts.numSubBatches
        batchStart = t + (labindex-1) + opts.batchSize ;
        batchEnd = min(t+2*opts.batchSize-1, numel(subset)) ;
      else
        batchStart = batchStart + numlabs ;
      end
      nextBatch = subset(batchStart : opts.numSubBatches * numlabs : batchEnd) ;
      getBatch(imdb, nextBatch) ;
    end

    if numGpus >= 1
      im = gpuArray(im) ;
    end

    % evaluate the CNN
    net.layers{end}.class = labels ;
    if training, dzdy = one; else, dzdy = [] ; end
    res = vl_simplenn(net, im, dzdy, res, ...
                      'accumulate', s ~= 1, ...
                      'mode', evalMode, ...
                      'conserveMemory', opts.conserveMemory, ...
                      'backPropDepth', opts.backPropDepth, ...
                      'sync', opts.sync, ...
                      'cudnn', opts.cudnn) ;

    % accumulate training errors
    error = sum([error, [...
      sum(double(gather(res(end).x))) ;
      reshape(opts.errorFunction(opts, labels, res),[],1) ; ]],2) ;
    numDone = numDone + numel(batch) ;
  end % next sub-batch

  % gather and accumulate gradients across labs
  if training
    if numGpus <= 1
      [net,res] = accumulate_gradients(opts, learningRate, batchSize, net, res) ;
    else
      if isempty(mmap)
        mmap = map_gradients(opts.memoryMapFile, net, res, numGpus) ;
      end
      write_gradients(mmap, net, res) ;
      labBarrier() ;
      [net,res] = accumulate_gradients(opts, learningRate, batchSize, net, res, mmap) ;
    end
  end

  % print learning statistics
  time = toc(start) ;
  stats = sum([stats,[0 ; error]],2); % works even when stats=[]
  stats(1) = time ;
  n = t + batchSize - 1 ; % number of images processed overall
  speed = n/time ;
  fprintf('%.1f Hz%s\n', speed) ;

  m = n / max(1,numlabs) ; % num images processed on this lab only
  fprintf(' obj:%.3g', stats(2)/m) ;
  for i=1:numel(opts.errorLabels)
    fprintf(' %s:%.3g', opts.errorLabels{i}, stats(i+2)/m) ;
  end
  fprintf(' [%d/%d]', numDone, batchSize);
  fprintf('\n') ;

  % debug info
  if opts.plotDiagnostics && numGpus <= 1
    figure(2) ; vl_simplenn_diagnose(net,res) ; drawnow ;
  end

end

if opts.profile
  if numGpus <= 1
    prof = profile('info') ;
    profile off ;
  else
    prof = mpiprofile('info');
    mpiprofile off ;
  end
else
  prof = [] ;
end

if numGpus >= 1
  net_cpu = vl_simplenn_move(net, 'cpu') ;
else
  net_cpu = net ;
end

% -------------------------------------------------------------------------
function [net,res] = accumulate_gradients(opts, lr, batchSize, net, res, mmap)
% -------------------------------------------------------------------------
if nargin >= 6
  numGpus = numel(mmap.Data) ;
else
  numGpus = 1 ;
end

for l=numel(net.layers):-1:1
  for j=1:numel(res(l).dzdw)

    % accumualte gradients from multiple labs (GPUs) if needed
    if numGpus > 1
      tag = sprintf('l%d_%d',l,j) ;
      tmp = zeros(size(mmap.Data(labindex).(tag)), 'single') ;
      for g = setdiff(1:numGpus, labindex)
        tmp = tmp + mmap.Data(g).(tag) ;
      end
      res(l).dzdw{j} = res(l).dzdw{j} + tmp ;
    end

    if j == 3 && strcmp(net.layers{l}.type, 'bnorm')
      % special case for learning bnorm moments
      thisLR = net.layers{l}.learningRate(j) ;
      net.layers{l}.weights{j} = ...
        (1-thisLR) * net.layers{l}.weights{j} + ...
        (thisLR/batchSize) * res(l).dzdw{j} ;
    else
      % standard gradient training
      thisDecay = opts.weightDecay * net.layers{l}.weightDecay(j) ;
      thisLR = lr * net.layers{l}.learningRate(j) ;
      net.layers{l}.momentum{j} = ...
        opts.momentum * net.layers{l}.momentum{j} ...
        - thisDecay * net.layers{l}.weights{j} ...
        - (1 / batchSize) * res(l).dzdw{j} ;
      net.layers{l}.weights{j} = net.layers{l}.weights{j} + ...
        thisLR * net.layers{l}.momentum{j} ;
    end
  end
end

% -------------------------------------------------------------------------
function mmap = map_gradients(fname, net, res, numGpus)
% -------------------------------------------------------------------------
format = {} ;
for i=1:numel(net.layers)
  for j=1:numel(res(i).dzdw)
    format(end+1,1:3) = {'single', size(res(i).dzdw{j}), sprintf('l%d_%d',i,j)} ;
  end
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
function write_gradients(mmap, net, res)
% -------------------------------------------------------------------------
for i=1:numel(net.layers)
  for j=1:numel(res(i).dzdw)
    mmap.Data(labindex).(sprintf('l%d_%d',i,j)) = gather(res(i).dzdw{j}) ;
  end
end

% -------------------------------------------------------------------------
function epoch = findLastCheckpoint(modelDir)
% -------------------------------------------------------------------------
list = dir(fullfile(modelDir, 'net-epoch-*.mat')) ;
tokens = regexp({list.name}, 'net-epoch-([\d]+).mat', 'tokens') ;
epoch = cellfun(@(x) sscanf(x{1}{1}, '%d'), tokens) ;
epoch = max([epoch 0]) ;
