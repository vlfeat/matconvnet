function [net, info] = cnn_train_mgpu(net, imdb, getBatch, varargin)
% CNN_TRAIN   Demonstrates training a CNN
%    CNN_TRAIN() is an example learner implementing stochastic gradient
%    descent with momentum to train a CNN for image classification.
%    It can be used with different datasets by providing a suitable
%    getBatch function.

opts.train = [] ;
opts.val = [] ;
opts.numEpochs = 300 ;
opts.batchSize = 256 ;
opts.gpus = [] ; % which GPU devices to use (none, one, or more)
opts.learningRate = 0.001 ;
opts.continue = false ;
opts.expDir = fullfile('data','exp') ;
opts.conserveMemory = false ;
opts.sync = true ;
opts.prefetch = false ;
opts.weightDecay = 0.0005 ;
opts.momentum = 0.9 ;
opts.errorType = 'multiclass' ;
opts.plotDiagnostics = false ;
opts.memoryMapFile = '/tmp/matconvnet.bin' ;
opts = vl_argparse(opts, varargin) ;

if ~exist(opts.expDir, 'dir'), mkdir(opts.expDir) ; end
if isempty(opts.train), opts.train = find(imdb.images.set==1) ; end
if isempty(opts.val), opts.val = find(imdb.images.set==2) ; end
if isnan(opts.train), opts.train = [] ; end

%opts.train = opts.train(1:10000) ;
%opts.val = opts.val(1:10000) ;

% -------------------------------------------------------------------------
%                                                    Network initialization
% -------------------------------------------------------------------------

for i=1:numel(net.layers)
  if ~strcmp(net.layers{i}.type,'conv'), continue; end
  net.layers{i}.filtersMomentum = zeros(size(net.layers{i}.filters), ...
    class(net.layers{i}.filters)) ;
  net.layers{i}.biasesMomentum = zeros(size(net.layers{i}.biases), ...
    class(net.layers{i}.biases)) ; %#ok<*ZEROLIKE>
  if ~isfield(net.layers{i}, 'filtersLearningRate')
    net.layers{i}.filtersLearningRate = 1 ;
  end
  if ~isfield(net.layers{i}, 'biasesLearningRate')
    net.layers{i}.biasesLearningRate = 1 ;
  end
  if ~isfield(net.layers{i}, 'filtersWeightDecay')
    net.layers{i}.filtersWeightDecay = 1 ;
  end
  if ~isfield(net.layers{i}, 'biasesWeightDecay')
    net.layers{i}.biasesWeightDecay = 1 ;
  end
end

numGpus = numel(opts.gpus) ;

% setup multiple GPUs
if numGpus > 1
  if isempty(gcp('nocreate')),
    parpool('local',numGpus) ;
    spmd, gpuDevice(opts.gpus(labindex)) ; end
  end
end
if exist(opts.memoryMapFile), delete(opts.memoryMapFile) ; end

% -------------------------------------------------------------------------
%                                                        Train and validate
% -------------------------------------------------------------------------

lr = 0 ;
for epoch=1:opts.numEpochs
  prevLr = lr ;
  lr = opts.learningRate(min(epoch, numel(opts.learningRate))) ;

  % fast-forward to where we stopped
  modelPath = @(ep) fullfile(opts.expDir, sprintf('net-epoch-%d.mat', ep));
  modelFigPath = fullfile(opts.expDir, 'net-train.pdf') ;
  if opts.continue
    if exist(modelPath(epoch),'file')
      if epoch == opts.numEpochs
        load(modelPath(epoch), 'net', 'info') ;
      end
      continue ;
    end
    if epoch > 1
      fprintf('resuming by loading epoch %d\n', epoch-1) ;
      load(modelPath(epoch-1), 'net', 'info') ;
    end
  end

  train = opts.train(randperm(numel(opts.train))) ;
  val = opts.val ;

  % reset momentum if needed
  if prevLr ~= lr
    fprintf('learning rate changed (%f --> %f): resetting momentum\n', prevLr, lr) ;
    net = reset_momentum(net) ;
  end

  % move to GPU as needed
  if numGpus == 1
    net = vl_simplenn_move(net, 'gpu') ;
  elseif numGpus > 1
    spmd(numGpus)
      net_ = vl_simplenn_move(net, 'gpu') ;
    end
  end

  % train and validate
  if numGpus <= 1
    [net,stats.train] = process_epoch(opts, getBatch, epoch, train, lr, imdb, net, true) ;
    [~,stats.val] = process_epoch(opts, getBatch, epoch, val, lr, imdb, net, false) ;
  else
    spmd(numGpus)
      [net_, stats_train_] = process_epoch(opts, getBatch, epoch, train, lr, imdb, net_, true) ;
      [~, stats_val_] = process_epoch(opts, getBatch, epoch, val, lr, imdb, net_, false) ;
    end
    stats.train = sum([stats_train_{:}],2) ;
    stats.val = sum([stats_val_{:}],2) ;
  end

  % save
  for f = {'train', 'val'}
    f = char(f) ;
    n = numel(eval(f)) ;
    info.(f).objective(epoch) = stats.(f)(1) / n ;
    info.(f).error(epoch) = stats.(f)(2) / n ;
    info.(f).topFiveError(epoch) = stats.(f)(3) / n ;
    info.(f).speed(epoch) = n / stats.(f)(4) ;
  end
  if numGpus > 1
    spmd(numGpus)
      net_ = vl_simplenn_move(net_, 'cpu') ;
    end
    net = net_{1} ;
  else
    net = vl_simplenn_move(net, 'cpu') ;
  end
  save(modelPath(epoch), 'net', 'info') ;

  figure(1) ; clf ;
  subplot(1,2,1) ;
  semilogy(1:epoch, info.train.objective, 'k') ; hold on ;
  semilogy(1:epoch, info.val.objective, 'b') ;
  xlabel('training epoch') ; ylabel('energy') ;
  grid on ;
  h=legend('train', 'val') ;
  set(h,'color','none');
  title('objective') ;
  subplot(1,2,2) ;
  switch opts.errorType
    case 'multiclass'
      plot(1:epoch, info.train.error, 'k') ; hold on ;
      plot(1:epoch, info.train.topFiveError, 'k--') ;
      plot(1:epoch, info.val.error, 'b') ;
      plot(1:epoch, info.val.topFiveError, 'b--') ;
      h=legend('train','train-5','val','val-5') ;
    case 'binary'
      plot(1:epoch, info.train.error, 'k') ; hold on ;
      plot(1:epoch, info.val.error, 'b') ;
      h=legend('train','val') ;
  end
  grid on ;
  xlabel('training epoch') ; ylabel('error') ;
  set(h,'color','none') ;
  title('error') ;
  drawnow ;
  print(1, modelFigPath, '-dpdf') ;
end

% -------------------------------------------------------------------------
function [err, err5] = compute_error(opts, labels, predictions)
% -------------------------------------------------------------------------
switch opts.errorType
  case 'multiclass'
    [~,predictions] = sort(predictions, 3, 'descend') ;
    error = ~bsxfun(@eq, predictions, reshape(labels, 1, 1, 1, [])) ;
    err = sum(sum(sum(error(:,:,1,:)))) ;
    err5 = sum(sum(sum(min(error(:,:,1:5,:),[],3)))) ;
  case 'binary'
    error = bsxfun(@times, predictions, labels) < 0 ;
    err = sum(error(:)) ;
    errt = 0 ;
end

% -------------------------------------------------------------------------
function  [net,stats,prof] = process_epoch(opts, getBatch, epoch, subset, lr, imdb, net, training)
% -------------------------------------------------------------------------

if training, mode = 'training' ; else, mode = 'validation' ; end
if nargout > 2, mpiprofile on ; end

numGpus = numel(opts.gpus) ;
if numGpus >= 1
  one = gpuArray(single(1)) ;
else
  one = single(1) ;
end
res = [] ;
mmap = [] ;
stats = zeros(4,1) ;

for t=1:opts.batchSize:numel(subset)
  % get next image batch and labels
  batch = subset(t:min(t+opts.batchSize-1, numel(subset))) ;
  batch_time = tic ;
  batchSize = numel(batch) ;
  fprintf('%s: epoch %02d: processing batch %3d of %3d ...', mode, epoch, ...
          fix(t/opts.batchSize)+1, ceil(numel(subset)/opts.batchSize)) ;

  % divide batches among labs (for multi GPU support)
  batch = batch(labindex:numlabs:end) ;
  if opts.prefetch
    nextBatch = subset(t+opts.batchSize:min(t+2*opts.batchSize-1, numel(subset))) ;
    nextBatch = nextBatch(labindex:numlabs:end) ;
  end

  % fetch and prefetch images
  [im, labels] = getBatch(imdb, batch) ;
  if opts.prefetch, getBatch(imdb, nextBatch) ; end
  if numGpus >= 1
    im = gpuArray(im) ;
  end

  % backprop
  net.layers{end}.class = labels ;
  if training
    res = vl_simplenn(net, im, one, res, ...
                      'conserveMemory', opts.conserveMemory, ...
                      'sync', opts.sync) ;
  else
    res = vl_simplenn(net, im, [], res, ...
                      'disableDropout', true, ...
                      'conserveMemory', opts.conserveMemory, ...
                      'sync', opts.sync) ;
  end
  [err,err5] = compute_error(opts, labels, gather(res(end-1).x)) ;
  error = [sum(double(gather(res(end).x))) ; err ; err5] ;

  % get and accumulate gradients
  if training
    if numGpus <= 1
      net = accumulate_gradients(opts, lr, batchSize, net, res) ;
    else
      if isempty(mmap)
        mmap = map_grads(opts.memoryMapFile, net, res, numGpus) ;
      end
      write_grads(mmap, net, res) ;
      labBarrier() ;
      [net,res] = accumulate_gradients(opts, lr, batchSize, net, res, mmap) ;
    end
  end

  % print information
  batch_time = toc(batch_time) ;
  stats = stats + [error ; batch_time] ;
  speed = batchSize/batch_time ;

  fprintf(' %.2f s (%.1f images/s)', batch_time, speed) ;
  n = (t + batchSize - 1) / max(1,numlabs) ;
  fprintf(' err %.1f err5 %.1f', stats(2)/n*100, stats(3)/n*100) ;
  fprintf(' [done %d of %d]', numel(batch), batchSize);
  fprintf('\n') ;

  % debug info
  if opts.plotDiagnostics && numGpus <= 1
    figure(2) ; vl_simplenn_diagnose(net,res) ; drawnow ;
  end
end

if nargout > 2
  prof = mpiprofile('info');
  mpiprofile off ;
end

% -------------------------------------------------------------------------
function net = reset_momentum(net)
% -------------------------------------------------------------------------
for l=1:numel(net.layers)
  if ~strcmp(net.layers{l}.type, 'conv'), continue ; end
  net.layers{l}.filtersMomentum = 0 * net.layers{l}.filtersMomentum ;
  net.layers{l}.biasesMomentum = 0 * net.layers{l}.biasesMomentum ;
end

% -------------------------------------------------------------------------
function [net,res] = accumulate_gradients(opts, lr, n, net, res, mmap)
% -------------------------------------------------------------------------
for l=1:numel(net.layers)
  if ~strcmp(net.layers{l}.type, 'conv'), continue ; end

  fd = opts.weightDecay * net.layers{l}.filtersWeightDecay ;
  bd = opts.weightDecay * net.layers{l}.biasesWeightDecay ;
  flr = lr * net.layers{l}.filtersLearningRate ;
  blr = lr * net.layers{l}.biasesLearningRate ;

  if nargin >= 6
    for j=1:2
      tag = sprintf('l%d_%d',l,j) ;
      tmp = zeros(size(mmap.Data(labindex).(tag)), 'single') ;
      for g = setdiff(1:numel(mmap.Data), labindex)
        tmp = tmp + mmap.Data(g).(tag) ;
      end
      res(l).dzdw{j} = res(l).dzdw{j} + tmp ;
    end
  end

  net.layers{l}.filtersMomentum = ...
      opts.momentum * net.layers{l}.filtersMomentum ...
      - flr * fd * net.layers{l}.filters ...
      - flr / n * res(l).dzdw{1} ;

  net.layers{l}.biasesMomentum = ...
      opts.momentum * net.layers{l}.biasesMomentum ...
      - blr * bd * net.layers{l}.biases ...
      - blr / n * res(l).dzdw{2} ;

  net.layers{l}.filters = net.layers{l}.filters + net.layers{l}.filtersMomentum ;
  net.layers{l}.biases = net.layers{l}.biases + net.layers{l}.biasesMomentum ;
end

% -------------------------------------------------------------------------
function mmap = map_grads(fname, net, res, numGpus)
% -------------------------------------------------------------------------
format = {} ;
for i=1:numel(net.layers)
  if ~strcmp(net.layers{i}.type, 'conv'), continue ; end
  for j=1:2
    format(end+1,1:3) = {...
      'single', ...
      size(res(i).dzdw{j}), ...
      sprintf('l%d_%d',i,j)} ;
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
function write_grads(mmap, net, res)
% -------------------------------------------------------------------------
for i=1:numel(net.layers)
  if ~strcmp(net.layers{i}.type, 'conv'), continue ; end
  for j=1:2
    mmap.Data(labindex).(sprintf('l%d_%d',i,j)) = ...
        gather(res(i).dzdw{j}) ;
  end
end
