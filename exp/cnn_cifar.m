function cnn_cifar

opts.dataDir = 'data/cifar' ;
opts.expDir = 'data/cifar-exp-1' ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.batchSize = 100 ;
opts.numEpochs = 100 ;
opts.continue = true ;
opts.useGpu = false ;
opts.learningRate = 0.001 ;
opts.learningRate = 0.0001 ;
opts.debug_weights = true;

run matlab/vl_setupnn ;

% --------------------------------------------------------------------
%                                                         Prepare data
% --------------------------------------------------------------------

imdb = getCifarImdb(opts) ;

% Define network, cifar10 quick
net.layers = {} ;
% 1 conv1
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', 1e-4*randn(5,5,3,32, 'single'), ...
                           'biases', zeros(1, 32, 'single'), ...
                           'lr', [1 2], ...
                           'stride', 1, ...
                           'pad', 2) ;
% 2 pool1 (max pool)
net.layers{end+1} = struct('type', 'maxpool', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', [0 1 0 1]) ;
% 3 relu1
net.layers{end+1} = struct('type', 'relu') ;
% 4 conv2
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', 0.01*randn(5,5,32,32, 'single'),...
                           'biases', zeros(1,32,'single'), ...
                           'lr', [1 2], ...
                           'stride', 1, ...
                           'pad', 2) ;
% 5 relu2
net.layers{end+1} = struct('type', 'relu') ;
% 6 pool2 (avg pool)
net.layers{end+1} = struct('type', 'conv', ... 
                           'filters', ones(3,3,1,32, 'single'),... 
                           'biases', zeros(1,32,'single'), ...
                           'lr', [0 0], ...
                           'stride', 2, ...
                           'pad', [0 1 0 1]) ;
% 7 conv3
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', 0.01*randn(5,5,32,64, 'single'),...
                           'biases', zeros(1,64,'single'), ...
                           'lr', [1 2], ...
                           'stride', 1, ...
                           'pad', 2) ;
% 8 relu3
net.layers{end+1} = struct('type', 'relu') ;
% 9 pool3 (avg pool)
net.layers{end+1} = struct('type', 'conv', ... 
                           'filters', ones(3,3,1,64, 'single'),... 
                           'biases', zeros(1,64,'single'), ...
                           'lr', [0 0], ...
                           'stride', 2, ...
                           'pad', [0 1 0 1]) ;
% 10 ip1
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', 0.1*randn(4,4,64,64, 'single'),...
                           'biases', zeros(1,64,'single'), ...
                           'lr', [1 2], ...
                           'stride', 1, ...
                           'pad', 0) ;
% 11 ip2
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', 0.1*randn(1,1,64,10, 'single'),...
                           'biases', zeros(1,10,'single'), ...
                           'lr', [1 2], ...
                           'stride', 1, ...
                           'pad', 0) ;
% 12 loss
net.layers{end+1} = struct('type', 'softmaxloss') ;

for i=1:numel(net.layers)
  if ~strcmp(net.layers{i}.type,'conv'), continue; end
  net.layers{i}.filtersMomentum = zeros('like',net.layers{i}.filters) ;
  net.layers{i}.biasesMomentum = zeros('like',net.layers{i}.biases) ;
end

if opts.useGpu
  net = vl_simplenn_move(net, 'gpu') ;
  for i=1:numel(net.layers)
    if ~strcmp(net.layers{i}.type,'conv'), continue; end
    net.layers{i}.filtersMomentum = gpuArray(net.layers{i}.filtersMomentum) ;
    net.layers{i}.biasesMomentum = gpuArray(net.layers{i}.biasesMomentum) ;
  end
end

% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------

train = find(imdb.images.set==1) ;
val = find(imdb.images.set==3) ;

info.train.objective = [] ;
info.train.error = [] ;
info.train.topFiveError = [] ;
info.val.objective = [] ;
info.val.error = [] ;
info.val.topFiveError = [] ;

if opts.useGpu
  imdb.images.data = gpuArray(imdb.images.data) ;
end

for epoch=1:opts.numEpochs
  train = train(randperm(numel(train))) ;

  % fast-forward to where we stopped
  modelPath = fullfile(opts.expDir, 'net-epoch-%d.mat') ;
  modelFigPath = fullfile(opts.expDir, 'net-train.pdf') ;
  if opts.continue && exist(sprintf(modelPath, epoch),'file'), continue ; end
  if opts.continue && epoch > 1 && exist(sprintf(modelPath, epoch-1), 'file')
    fprintf('resuming from loading epoch %d\n', epoch-1) ;
    load(sprintf(modelPath, epoch-1), 'net', 'info') ;
  end

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
    im = imdb.images.data(:,:,:,batch) ;
    labels = imdb.images.labels(1,batch) ;

    % backprop
    net.layers{end}.class = labels ;
    res = vl_simplenn(net, im, 1) ;

    % update energy
    info.train.objective(end) = info.train.objective(end) + double(gather(res(end).x)) ;
    [~,predictions] = sort(squeeze(res(end-1).x), 'descend') ;
    error = ~bsxfun(@eq, predictions, labels) ;
    info.train.error(end) = info.train.error(end) + sum(error(1,:)) ;
    info.train.topFiveError(end) = info.train.topFiveError(end) + sum(min(error(1:5,:))) ;

    % gradient step
    for l=1:numel(net.layers)
      ly = net.layers{l} ;
      if ~strcmp(ly.type, 'conv'), continue ; end

      ly.filtersMomentum = 0.9 * ly.filtersMomentum ...
        - 0.0005 * opts.learningRate * ly.filters ...
        - opts.learningRate*ly.lr(1)/numel(batch) * res(l).dzdw{1} ;

      ly.biasesMomentum = 0.9 * ly.biasesMomentum ...
        - 0.0005 * opts.learningRate * ly.biases ...
        - opts.learningRate*ly.lr(2)/numel(batch) * res(l).dzdw{2} ;

      ly.filters = ly.filters + ly.filtersMomentum ;
      ly.biases = ly.biases + ly.biasesMomentum ;
      net.layers{l} = ly ;
    end

    if opts.debug_weights && mod(t-1,10*opts.batchSize)==0
      figure(100) ; clf ;
      n=numel(net.layers)+1 ;
      for l=1:n
        subplot(4,n,l) ;
        hist(res(l).x(:)) ;
        title(sprintf('layer %d input', l)) ;
        subplot(4,n,l+n) ;
        hist(res(l).dzdx(:)) ;
        title(sprintf('layer %d input der', l)) ;
        if l < n && isfield(net.layers{l}, 'filters')
          subplot(4,n,l+2*n) ;
          hist(net.layers{l}.filters(:)) ;
          title(sprintf('layer %d filters', l)) ;
          subplot(4,n,l+3*n) ;
          hist(res(l).dzdw{1}(:)) ;
          title(sprintf('layer %d filters der', l)) ;
        end
      end
      drawnow ;
    end

    batch_time = toc(batch_time) ;
    fprintf(' %.2f s (%.1f images/s)\n', batch_time, numel(batch)/ batch_time) ;
  end % next batch

  % evaluation on validation set
  for t=1:opts.batchSize:numel(val)
    batch = val(t:min(t+opts.batchSize-1, numel(val))) ;
    fprintf('validation: epoch %02d: processing batch %3d of %3d\n', epoch, ...
      fix(t/opts.batchSize)+1, ceil(numel(val)/opts.batchSize)) ;
    im = imdb.images.data(:,:,:,batch) ;
    labels = imdb.images.labels(1,batch) ;

    net.layers{end}.class = labels ;
    res = vl_simplenn(net, im) ;

    % update energy
    info.val.objective(end) = info.val.objective(end) + double(gather(res(end).x)) ;
    [~,predictions] = sort(squeeze(res(end-1).x), 'descend') ;
    error = ~bsxfun(@eq, predictions, labels) ;
    info.val.error(end) = info.val.error(end) + sum(error(1,:)) ;
    info.val.topFiveError(end) = info.val.topFiveError(end) + sum(min(error(1:5,:))) ;
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
  subplot(2,2,1) ;
  semilogy(1:epoch, info.train.objective, 'k') ; hold on ;
  semilogy(1:epoch, info.val.objective, 'b') ;
  xlabel('epoch') ; ylabel('energy') ; legend('train', 'val') ; grid on ;
  title('objective') ;
  subplot(2,2,2) ;
  plot(1:epoch, info.train.error, 'k') ; hold on ;
  plot(1:epoch, info.train.topFiveError, 'k--') ;
  plot(1:epoch, info.val.error, 'b') ;
  plot(1:epoch, info.val.topFiveError, 'b--') ;
  xlabel('epoch') ; ylabel('energy') ; legend('train','train-5','val','val-5') ; grid on ;
  title('error') ;
  subplot(2,2,3) ;
  vl_imarraysc(squeeze(net.layers{1}.filters),'spacing',2) ;
  axis equal ;
  colormap gray ;
  drawnow ;
  print(1, modelFigPath, '-dpdf') ;
end


% --------------------------------------------------------------------
function imdb = getCifarImdb(opts)
% --------------------------------------------------------------------

unpackPath = fullfile(opts.dataDir, 'cifar-10-batches-mat');
files = [arrayfun(@(n) sprintf('data_batch_%d.mat', n), 1:5, 'UniformOutput', false) ...
  {'test_batch.mat'}];
files = cellfun(@(fn) fullfile(unpackPath, fn), files, 'UniformOutput', false);
file_set = uint8([ones(1, 5), 3]);

mkdir(opts.dataDir) ;
if any(cellfun(@(fn) ~exist(fn, 'file'), files))
  url = 'http://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz' ;
  fprintf('downloading %s\n', url) ;
  untar(url, opts.dataDir) ;
end

data = cell(1, numel(files));
labels = cell(1, numel(files));
sets = cell(1, numel(files));
for fi = 1:numel(files)
  fd = load(files{fi}) ;
  data{fi} = permute(reshape(fd.data',32,32,3,[]),[2 1 3 4]) ;
  labels{fi} = fd.labels' + 1; % Index from 1
  sets{fi} = repmat(file_set(fi), size(labels{fi}));
end

data = single(cat(4, data{:}));
dataMean = mean(data, 4);
data = bsxfun(@minus, data, dataMean);

clNames = load(fullfile(unpackPath, 'batches.meta.mat'));

imdb.images.data = data ;
imdb.images.data_mean = dataMean;
imdb.images.labels = single(cat(2, labels{:})) ;
imdb.images.set = cat(2, sets{:});
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.classes = clNames.label_names;
