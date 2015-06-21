function [net, info] = cnn_dagtrain(net, imdb, getBatch, outputDefs, varargin)
% CNN_TRAIN   Demonstrates training a CNN
%    CNN_TRAIN() is an example learner implementing stochastic gradient
%    descent with momentum to train a CNN for image classification.
%    It can be used with different datasets by providing a suitable
%    getBatch function.

opts.numEpochs = 300 ;
opts.batchSize = 256 ;
opts.useGpu = false ; 
opts.learningRate = 0.001 ;
opts.continue = false ;
opts.expDir = fullfile('data','exp') ;
opts.conserveMemory = false ;
opts.prefetch = false ;
opts.weightDecay = 0.0005 ;
opts.addRegul = [];
opts.momentum = 0.9 ;
opts.plotDiagnostics = false ;
opts.save = true;
opts.plotResultsFun = ...
  @(info, opts) cnn_dagtrain_plotmcress(info, opts, 'logobjplot', false);
opts.epochSize = inf;
opts.randseed = 0;
opts.preloadEpoch = [];
opts.getTrain = @(imdb, epoch) randsample(find(imdb.images.set==1), ...
  sum(imdb.images.set==1));
opts.getVal = @(imdb) find(imdb.images.set==2);
opts.getNumSamples = @(opts, input) opts.batchSize;
opts.trainHist = []; % When set, collect histograms of the training data seen

opts = vl_argparse(opts, varargin) ;

if ~exist(opts.expDir, 'dir'), mkdir(opts.expDir) ; end

% -------------------------------------------------------------------------
%                                                    Network initialization
% -------------------------------------------------------------------------

for i=1:numel(net.layers)
  if ~ismember(net.layers{i}.type, {'conv', 'bnorm'}), continue; end
  net.layers{i}.filtersMomentum = zeros('like',net.layers{i}.weights{1}) ;
  net.layers{i}.biasesMomentum = zeros('like',net.layers{i}.weights{2}) ;
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
    % TODO nasty solution
    if ~ismember(net.layers{i}.type, {'conv', 'bnorm'}), continue; end
    net.layers{i}.filtersMomentum = gpuArray(net.layers{i}.filtersMomentum) ;
    net.layers{i}.biasesMomentum = gpuArray(net.layers{i}.biasesMomentum) ;
  end
else
  net = vl_simplenn_move(net, 'cpu');
end

% -------------------------------------------------------------------------
%                                                        Train and validate
% -------------------------------------------------------------------------

rng(opts.randseed) ;

if opts.useGpu
  one = gpuArray(single(1)) ;
else
  one = single(1) ;
end
dzdy = struct('name',{outputDefs.layerName},'dzdx',one);

info.train.lr = [] ;
info.val = [];
if ~isempty(opts.trainHist), info.train.hist = opts.trainHist; end

val = [];
if ~isempty(opts.getVal)
  val = opts.getVal(imdb) ;
end

lr = 0 ;
model_loaded = false;
arcs = [];
for epoch=1:opts.numEpochs
  prevLr = lr ;
  
  % fast-forward to where we stopped
  modelPath = fullfile(opts.expDir, 'net-epoch-%d.mat') ;
  if opts.continue
    if exist(sprintf(modelPath, epoch),'file'), continue ; end
    if epoch > 1
      fprintf('resuming by loading epoch %d\n', epoch-1) ;
      load(sprintf(modelPath, epoch-1), 'net', 'info') ;
      if ~isempty(info.train.lr)
        lr = info.train.lr(end);
      end
    end
  end
  
  % Set the learning rate
  lr = opts.learningRate(min(epoch, numel(opts.learningRate))) ;
  info.train.lr(end+1) = lr;
  model_loaded = true;
  
  train = [];
  if ~isempty(opts.getTrain)
    train = opts.getTrain(imdb, epoch);
  end
  
  if ~isempty(opts.trainHist)
    info.train.hist(train) = info.train.hist(train) + 1;
  end
  
  if ~isempty(opts.preloadEpoch)
    to_load = [train, val];
    fprintf('Preloading %d dpoints... ', numel(to_load)); stime = tic;
    imdb = opts.preloadEpoch(imdb, to_load);
    fprintf('Done %.2fs\n', toc(stime));
  end;

  % reset momentum if needed
  if prevLr ~= lr
    fprintf('learning rate changed (%f --> %f): resetting momentum\n', prevLr, lr) ;
    for l=1:numel(net.layers)
      ly = net.layers{l} ;
      if ~ismember(ly.type, {'conv', 'bnorm'}), continue ; end
      ly.filtersMomentum = 0 * ly.filtersMomentum ;
      ly.biasesMomentum = 0 * ly.biasesMomentum ;
    end
  end

  step = 1;
  nsamples_seen_train = 0;
  for t=1:opts.batchSize:numel(train)
    if isnan(train), break; end;
    % deltes all allocated buffers
    clear res ;

    % get next image batch and labels
    batch = train(t:min(t+opts.batchSize-1, numel(train))) ;
    batch_time = tic ;
    fprintf('training: epoch %02d: batch %3d of %3d ...', epoch, ...
            fix(t/opts.batchSize)+1, ceil(numel(train)/opts.batchSize)) ;

    input = getBatch(imdb, batch) ;
    if isempty(input)
      warning('Invalid batch.');
      continue;
    end;
    nsamples = opts.getNumSamples(opts, input);

    if isempty(arcs) || ~isfield(outputDefs, 'bidxs')
      arcs = vl_dagnn_getarcs(net, input);
      % TODO solve more elegantly
      [~, bufferIdxs] = ismember({outputDefs.layerName}, arcs.bufferNames);
      bufferIdxs = num2cell(bufferIdxs);
      [outputDefs(:).bidxs] = deal(bufferIdxs{:});
    end
    
    if opts.useGpu
      for fi = 1:numel(input), input(fi).x = gpuArray(input(fi).x); end
    end
    
    if opts.prefetch
      nextBatch = train(t+opts.batchSize:min(t+2*opts.batchSize-1, numel(train))) ;
      getBatch(imdb, nextBatch) ;
    end

    % backprop
    [res, dzdw] = vl_dagnn(net, input, dzdy, [], arcs, ...
      'disableDropout', false) ;
    
    info.train = updateOutputs(info.train, res, outputDefs, epoch) ;

    % gradient step
    for l=1:numel(net.layers)
      if isempty(dzdw{l}), continue ; end
      flr = net.layers{l}.filtersLearningRate;
      blr = net.layers{l}.biasesLearningRate;
      if flr == 0 && blr == 0, continue; end;
      
      net.layers{l}.filtersMomentum = ...
        opts.momentum * net.layers{l}.filtersMomentum ...
          - (lr * flr) * ...
          (opts.weightDecay * net.layers{l}.filtersWeightDecay) * net.layers{l}.weights{1} ...
          - (lr * flr) / nsamples * dzdw{l}{1} ;

      net.layers{l}.biasesMomentum = ...
        opts.momentum * net.layers{l}.biasesMomentum ...
          - (lr * blr) * ....
          (opts.weightDecay * net.layers{l}.biasesWeightDecay) * net.layers{l}.weights{2} ...
          - (lr * blr) / nsamples * dzdw{l}{2} ;

      net.layers{l}.weights{1} = net.layers{l}.weights{1} + net.layers{l}.filtersMomentum ;
      net.layers{l}.weights{2} = net.layers{l}.weights{2} + net.layers{l}.biasesMomentum ;
    end
    
    % print information
    batch_time = toc(batch_time) ;
    fprintf(' %.2f s (%.1f images/s) ', batch_time, nsamples/batch_time) ;
    nsamples_seen_train = nsamples_seen_train + nsamples;
    
    % Print the output if necessary
    if isfield(outputDefs, 'printfun')
      for oi = 1:numel(outputDefs)
        if ~isempty(outputDefs(oi).printfun)
          outputDefs(oi).printfun(...
            info.train.(outputDefs(oi).name)(:,epoch)/nsamples_seen_train) ;
        end
      end
    end
    fprintf('\n') ;

    if ~isempty(opts.netcheckFun)
      [has_exploded, he_state] = opts.netcheckFun(net, info, he_state);
      if has_exploded
        warning('!!! NETWORK EXPLODED');
        info.exploded = true;
        return;
      end
    end
    
    % debug info
    if opts.plotDiagnostics
      figure(2) ; vl_simplenn_diagnose(net,res) ; drawnow ;
    end
    step = step+1;
  end % next batch

  % evaluation on validation set
  bi = 1;
  nsamples_seen_val = 0;
  for t=1:opts.batchSize:numel(val)
    batch_time = tic ;
    batch = val(t:min(t+opts.batchSize-1, numel(val))) ;
    fprintf('validation: epoch %02d: batch %3d of %3d ...', epoch, ...
            fix(t/opts.batchSize)+1, ceil(numel(val)/opts.batchSize)) ;
    input = getBatch(imdb, batch) ;
    if isempty(input)
      warning('Invalid batch.');
      continue;
    end;
    nsamples = opts.getNumSamples(opts, input);
    
    if isempty(arcs) || ~isfield(outputDefs, 'bidxs')
      arcs = vl_dagnn_getarcs(net, input);
      % TODO solve more elegantly
      [~, bufferIdxs] = ismember({outputDefs.layerName}, arcs.bufferNames);
      bufferIdxs = num2cell(bufferIdxs);
      [outputDefs(:).bidxs] = deal(bufferIdxs{:});
    end
    
    if opts.prefetch
      nextBatch = val(t+opts.batchSize:min(t+2*opts.batchSize-1, numel(val))) ;
      getBatch(imdb, nextBatch) ;
    end
    if opts.useGpu
      for fi = 1:numel(input), input(fi).x = gpuArray(input(fi).x); end
    end

    clear res ;
    res = vl_dagnn(net, input, [], [], arcs, 'disableDropout', true) ;
    
    nsamples_seen_val = nsamples_seen_val + nsamples;
    batch_time = toc(batch_time) ;
    fprintf(' %.2f s (%.1f images/s)', batch_time, nsamples/batch_time) ;
    
    info.val = updateOutputs(info.val, res, outputDefs, epoch);
    % Print the output if necessary
    if isfield(outputDefs, 'printfun')
      for oi = 1:numel(outputDefs)
        if ~isempty(outputDefs(oi).printfun)
          outputDefs(oi).printfun(...
            info.val.(outputDefs(oi).name)(:,end)/nsamples_seen_val) ;
        end
      end
    end
    fprintf('\n') ;
    bi = bi + 1;
  end
 
  for oi = 1:numel(outputDefs)
    if train > 0
      info.train.(outputDefs(oi).name)(:,end) = ...
        info.train.(outputDefs(oi).name)(:,end) / nsamples_seen_train;
    end
    if val > 0
      info.val.(outputDefs(oi).name)(:,end) = ...
        info.val.(outputDefs(oi).name)(:,end) / nsamples_seen_val;
    end
  end
  
  % Save
  if opts.save
    save(sprintf(modelPath,epoch), 'net', 'info') ;
  end

  if ~isempty(opts.plotResultsFun)
    opts.plotResultsFun(info, opts);
  end
end
if ~model_loaded
  fprintf('Loading epoch %d\n', epoch) ;
  load(sprintf(modelPath, epoch), 'net', 'info') ;
end

% -------------------------------------------------------------------------
function nsamples = getNumSamples(opts, input)
% -------------------------------------------------------------------------
ni = numel(input) - 2;
if ndims(input(ni).x) == 4
  nsamples = size(input(ni).x,4);
else
  error('Unable to get the number of samples.');
end

% -------------------------------------------------------------------------
function info = updateOutputs(info, res, outputDefs, epoch)
% -------------------------------------------------------------------------
assert(isfield(outputDefs, 'bidxs'));
for oii = 1:numel(outputDefs)
  value = gather(res(outputDefs(oii).bidxs).x);
  if ~isfield(info, outputDefs(oii).name) || ...
      size(info.(outputDefs(oii).name), 2) < epoch
    prev_value = zeros(numel(value), 1);
  else
    prev_value = info.(outputDefs(oii).name)(:,epoch);
  end
  info.(outputDefs(oii).name)(:,epoch) = prev_value + value;
end


