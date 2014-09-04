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
opts.weightDecay = 0.0005 ;
opts.momentum = 0.9 ;
opts.errorType = 'multiclass' ;
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
  if opts.continue & exist(sprintf(modelPath, epoch),'file'), continue ; end
  if opts.continue & epoch > 1 & exist(sprintf(modelPath, epoch-1), 'file')
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
    clear res ;
    net.layers{end}.class = labels ;
    res = vl_simplenn(net, im, one, 'conserveMemory', opts.conserveMemory) ;
    info.train = updateError(opts, info.train, net, res) ;

    % gradient step
    lr = opts.learningRate(min(epoch, numel(opts.learningRate))) ;
    for l=1:numel(net.layers)
      ly = net.layers{l} ;
      if ~strcmp(ly.type, 'conv'), continue ; end

      ly.filtersMomentum = opts.momentum * ly.filtersMomentum ...
          - opts.weightDecay * ly.filtersWeightDecay ...
              * lr * ly.filtersLearningRate * ly.filters ...
          - lr * ly.filtersLearningRate/numel(batch) * res(l).dzdw{1} ;

      ly.biasesMomentum = opts.momentum * ly.biasesMomentum ...
          - opts.weightDecay * ly.biasesWeightDecay ...
              * lr * ly.biasesLearningRate * ly.biases ...
          - lr * ly.biasesLearningRate/numel(batch) * res(l).dzdw{2} ;

      ly.filters = ly.filters + ly.filtersMomentum ;
      ly.biases = ly.biases + ly.biasesMomentum ;
      net.layers{l} = ly ;
    end

    % print information
    batch_time = toc(batch_time) ;
    fprintf(' %.2f s (%.1f images/s)', batch_time, numel(batch)/batch_time) ;
    n = t + numel(batch) - 1 ;
    fprintf(' err %.1f err5 %.1f', ...
      info.train.error(end)/n*100, info.train.topFiveError(end)/n*100) ;
    fprintf('\n') ;
    
    if 0
      diagnose(net,res) ;
    end
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
    info.val = updateError(opts, info.val, net, res) ;

    % print information
    batch_time = toc(batch_time) ;
    fprintf(' %.2f s (%.1f images/s)', batch_time, numel(batch)/batch_time) ;
    n = t + numel(batch) - 1 ;
    fprintf(' err %.1f err5 %.1f', ...
      info.val.error(end)/n*100, info.val.topFiveError(end)/n*100) ;
    fprintf('\n') ;
  end

  % save
  info.train.objective(end) = info.train.objective(end) / numel(train) ;
  info.train.error(end) = info.train.error(end) / numel(train)  ;
  info.train.topFiveError(end) = info.train.topFiveError(end) / numel(train) ;
  info.val.objective(end) = info.val.objective(end) / numel(val) ;
  info.val.error(end) = info.val.error(end) / numel(val) ;
  info.val.topFiveError(end) = info.val.topFiveError(end) / numel(val) ;
  save(sprintf(modelPath,epoch), 'net', 'info') ;

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
function info = updateError(opts, info, net, res)
% -------------------------------------------------------------------------
predictions = gather(res(end-1).x) ;
sz = size(predictions) ;
n = prod(sz(1:2)) ;
    
labels = net.layers{end}.class ;
info.objective(end) = info.objective(end) + sum(double(gather(res(end).x))) ;
switch opts.errorType
  case 'multiclass'
    if any(sz>1)
      warning('mutliclass loss does cannot yet be applied in a convolutional manner') ;
    end
    [~,predictions] = sort(squeeze(predictions), 'descend') ;
    error = ~bsxfun(@eq, predictions, labels) ;
    info.error(end) = info.error(end) + sum(error(1,:)) ;
    info.topFiveError(end) = info.topFiveError(end) + sum(min(error(1:5,:))) ;
  case 'binary'
    error = bsxfun(@times, predictions, labels) < 0 ;
    info.error(end) = info.error(end) + sum(error(:))/n ;
end

% -------------------------------------------------------------------------
function diagnose(net, res)
% -------------------------------------------------------------------------
n = numel(net.layers) ;
fmu = NaN + zeros(1, n) ;
fmi = fmu ;
fmx = fmu ;
bmu = fmu ;
bmi = fmu ;
bmx = fmu ;
xmu = fmu ;
xmi = fmi ;
xmx = fmx ;
dxmu = fmu ;
dxmi = fmi ;
dxmx = fmx ;
dfmu = fmu ;
dfmi = fmu ;
dfmx = fmu ;
dbmu = fmu ;
dbmi = fmu ;
dbmx = fmu ;

for i=1:numel(net.layers)
  ly = net.layers{i} ;
  if strcmp(ly.type, 'conv') && numel(ly.filters) > 0
    x = gather(ly.filters) ;
    fmu(i) = mean(x(:)) ;
    fmi(i) = min(x(:)) ;
    fmx(i) = max(x(:)) ;
  end
  if strcmp(ly.type, 'conv') && numel(ly.biases) > 0
    x = gather(ly.biases) ;
    bmu(i) = mean(x(:)) ;
    bmi(i) = min(x(:)) ;
    bmx(i) = max(x(:)) ;
  end
  if numel(res(i).x) > 1
    x = gather(res(i).x) ;
    xmu(i) = mean(x(:)) ;
    xmi(i) = min(x(:)) ;
    xmx(i) = max(x(:)) ;
  end
  if numel(res(i).dzdx) > 1
    x = gather(res(i).dzdx);
    dxmu(i) = mean(x(:)) ;
    dxmi(i) = min(x(:)) ;
    dxmx(i) = max(x(:)) ;
  end
  if strcmp(ly.type, 'conv') && numel(res(i).dzdw{1}) > 0
    x = gather(res(i).dzdw{1}) ;
    dfmu(i) = mean(x(:)) ;
    dfmi(i) = min(x(:)) ;
    dfmx(i) = max(x(:)) ;
  end
  if strcmp(ly.type, 'conv') && numel(res(i).dzdw{2}) > 0
    x = gather(res(i).dzdw{2}) ;
    dbmu(i) = mean(x(:)) ;
    dbmi(i) = min(x(:)) ;
    dbmx(i) = max(x(:)) ;
  end
end

figure(2) ; clf ;
subplot(6,1,1) ;
errorbar(1:n, fmu, fmi, fmx, 'bo') ;
grid on ;
xlabel('layer') ;
ylabel('filters') ;

subplot(6,1,2) ;
errorbar(1:n, bmu, bmi, bmx, 'bo') ;
grid on ;
xlabel('layer') ;
ylabel('biases') ;

subplot(6,1,3) ;
errorbar(1:n, xmu, xmi, xmx, 'bo') ;
grid on ;
xlabel('layer') ;
ylabel('x') ;

subplot(6,1,4) ;
errorbar(1:n, dxmu, dxmi, dxmx, 'bo') ;
grid on ;
xlabel('layer') ;
ylabel('dzdx') ;

subplot(6,1,5) ;
errorbar(1:n, dfmu, dfmi, dfmx, 'bo') ;
grid on ;
xlabel('layer') ;
ylabel('dfilters') ;

subplot(6,1,6) ;
errorbar(1:n, dbmu, dbmi, dbmx, 'bo') ;
grid on ;
xlabel('layer') ;
ylabel('dbiases') ;

title('coefficient ranges') ;
drawnow ;


