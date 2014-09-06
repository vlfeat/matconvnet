function [net, info] = cnn_train(net, imdb, getBatch, varargin)
% CNN_TRAIN   Demonstrates training a CNN
%    CNN_TRAIN() is an example learner implementing stochastic gradient
%    descent with momentum to train a CNN for image classification.
%    It can be used with different datasets by providing a suitable
%    getBatch function.

opts.train = [] ;
opts.val = [] ;
opts.numEpochs = 300 ;
opts.batchSize = 256 ;
opts.useGpu = false ;
opts.learningRate = 0.001 ;
opts.continue = false ;
opts.expDir = 'data/exp' ;
opts.conserveMemory = false ;
opts.prefetch = false ;
opts.weightDecay = 0.0005;
opts.momentum = 0.9;
opts = vl_argparse(opts, varargin) ;

if ~exist(opts.expDir), mkdir(opts.expDir) ; end
if isempty(opts.train), opts.train = find(imdb.images.set==1) ; end
if isempty(opts.val), opts.val = find(imdb.images.set==2) ; end
if isnan(opts.train), opts.train = [] ; end

% -------------------------------------------------------------------------
%                                                    Network initialization
% -------------------------------------------------------------------------

for i=1:numel(net.layers)
  if ~strcmp(net.layers{i}.type,'conv'), continue; end
  net.layers{i}.filtersMomentum = zeros('like',net.layers{i}.filters) ;
  net.layers{i}.biasesMomentum = zeros('like',net.layers{i}.biases) ;
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

if opts.useGpu
  net = vl_simplenn_move(net, 'gpu') ;
  for i=1:numel(net.layers)
    if ~strcmp(net.layers{i}.type,'conv'), continue; end
    net.layers{i}.filtersMomentum = gpuArray(net.layers{i}.filtersMomentum) ;
    net.layers{i}.biasesMomentum = gpuArray(net.layers{i}.biasesMomentum) ;
  end
end

% -------------------------------------------------------------------------
%                                                        Train and validate
% -------------------------------------------------------------------------
rng(0) ;
if opts.useGpu
  one = gpuArray(single(1)) ;
else
  one = single(1) ;
end

info.train.objective = [] ;
info.train.error = [] ;
info.train.topFiveError = [] ;
info.val.objective = [] ;
info.val.error = [] ;
info.val.topFiveError = [] ;

for epoch=1:opts.numEpochs
  % fast-forward to where we stopped
  modelPath = fullfile(opts.expDir, 'net-epoch-%d.mat') ;
  modelFigPath = fullfile(opts.expDir, 'net-train.pdf') ;
  if opts.continue
    if exist(sprintf(modelPath, epoch),'file'), continue ; end
    fprintf('resuming by loading epoch %d\n', epoch-1) ;
    load(sprintf(modelPath, epoch-1), 'net', 'info') ;
  end

  train = opts.train(randperm(numel(opts.train))) ;
  val = opts.val ;

  info.train.objective(end+1) = 0 ;
  info.train.error(end+1) = 0 ;
  info.train.topFiveError(end+1) = 0 ;
  info.val.objective(end+1) = 0 ;
  info.val.error(end+1) = 0 ;
  info.val.topFiveError(end+1) = 0 ;

  for t=1:opts.batchSize:numel(train)
    % get next image batch and labels
    batch = train(t:min(t+opts.batchSize-1, numel(train))) ;
    batch_time = tic ;
    fprintf('training: epoch %02d: processing batch %3d of %3d ...', epoch, ...
            fix(t/opts.batchSize)+1, ceil(numel(train)/opts.batchSize)) ;
    [im, labels] = getBatch(imdb, batch) ;
    if opts.prefetch
      nextBatch = train(t+opts.batchSize:min(t+2*opts.batchSize-1, numel(train))) ;
      getBatch(imdb, nextBatch) ;
    end
    if opts.useGpu
      im = gpuArray(im) ;
    end

    % backprop
    net.layers{end}.class = labels ;
    clear res;
    res = vl_simplenn(net, im, one, 'conserveMemory', opts.conserveMemory) ;

    % update energy
    info.train.objective(end) = info.train.objective(end) + double(gather(res(end).x)) ;
    [~,predictions] = sort(squeeze(gather(res(end-1).x)), 'descend') ;
    error = ~bsxfun(@eq, predictions, labels) ;
    info.train.error(end) = info.train.error(end) + sum(error(1,:)) ;
    info.train.topFiveError(end) = info.train.topFiveError(end) + sum(min(error(1:5,:))) ;

    % gradient step
    lr = opts.learningRate(min(epoch, numel(opts.learningRate))) ;
    for l=1:numel(net.layers)
      ly = net.layers{l} ;
      if ~strcmp(ly.type, 'conv'), continue ; end

      ly.filtersMomentum = ...
        opts.momentum * ly.filtersMomentum ...
          - (lr * ly.filtersLearningRate) * (opts.weightDecay * ly.filtersWeightDecay) * ly.filters ...
          - (lr * ly.filtersLearningRate) / numel(batch) * res(l).dzdw{1} ;

      ly.biasesMomentum = ...
        opts.momentum * ly.biasesMomentum ...
          - (lr * ly.biasesLearningRate) * (opts.weightDecay * ly.biasesWeightDecay) * ly.biases ...
          - (lr * ly.biasesLearningRate) / numel(batch) * res(l).dzdw{2} ;

      ly.filters = ly.filters + ly.filtersMomentum ;
      ly.biases = ly.biases + ly.biasesMomentum ;
      net.layers{l} = ly ;
    end

    batch_time = toc(batch_time) ;
    fprintf(' %.2f s (%.1f images/s)\n', batch_time, numel(batch)/batch_time) ;
  end % next batch

  % evaluation on validation set
  for t=1:opts.batchSize:numel(val)
    batch_time = tic ;
    batch = val(t:min(t+opts.batchSize-1, numel(val))) ;
    fprintf('validation: epoch %02d: processing batch %3d of %3d ...', epoch, ...
            fix(t/opts.batchSize)+1, ceil(numel(val)/opts.batchSize)) ;
    [im, labels] = getBatch(imdb, batch) ;
    if opts.prefetch
      nextBatch = val(t+opts.batchSize:min(t+2*opts.batchSize-1, numel(val))) ;
      getBatch(imdb, nextBatch) ;
    end
    if opts.useGpu
      im = gpuArray(im) ;
    end

    net.layers{end}.class = labels ;
    res = vl_simplenn(net, im, [], 'disableDropout', true) ;

    % update energy
    info.val.objective(end) = info.val.objective(end) + double(gather(res(end).x)) ;
    [~,predictions] = sort(squeeze(gather(res(end-1).x)), 'descend') ;
    error = ~bsxfun(@eq, predictions, labels) ;
    info.val.error(end) = info.val.error(end) + sum(error(1,:)) ;
    info.val.topFiveError(end) = info.val.topFiveError(end) + sum(min(error(1:5,:))) ;

    batch_time = toc(batch_time) ;
    fprintf(' %.2f s (%.1f images/s)', batch_time, numel(batch)/ batch_time) ;
    if 1
      n = t + numel(batch) - 1 ;
      fprintf(' err %.1f err5 %.1f', ...
        info.val.error(end)/n*100, info.val.topFiveError(end)/n*100) ;
    end
    if opts.useGpu
      %gpu = gpuDevice ;
      %fprintf(' [GPU free memory %.2fMB]', gpu.FreeMemory/1024^2) ;
    end
    fprintf('\n') ;
  end

  % save
  info.train.objective(end) = info.train.objective(end) / numel(train) ;
  info.train.error(end) = info.train.error(end) / numel(train)  ;
  info.train.topFiveError(end) = info.train.topFiveError(end) / numel(train) ;
  info.val.objective(end) = info.val.objective(end) / numel(val) ;
  info.val.error(end) = info.val.error(end) / numel(val)  ;
  info.val.topFiveError(end) = info.val.topFiveError(end) / numel(val) ;
  save(sprintf(modelPath,epoch), 'net', 'info') ;

  figure(1) ; clf ;
  subplot(1,2,1) ;
  semilogy(1:epoch, info.train.objective, 'k') ; hold on ;
  semilogy(1:epoch, info.val.objective, 'b') ;
  xlabel('epoch') ; ylabel('energy') ; h=legend('train', 'val') ; grid on ;
  set(h,'color','none');
  title('objective') ;
  subplot(1,2,2) ;
  plot(1:epoch, info.train.error, 'k') ; hold on ;
  plot(1:epoch, info.train.topFiveError, 'k--') ;
  plot(1:epoch, info.val.error, 'b') ;
  plot(1:epoch, info.val.topFiveError, 'b--') ;
  xlabel('epoch') ; ylabel('energy') ; h=legend('train','train-5','val','val-5') ; grid on ;
  set(h,'color','none') ;
  title('error') ;
  drawnow ;
  print(1, modelFigPath, '-dpdf') ;
end
