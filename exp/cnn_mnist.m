function cnn_mnist

opts.dataDir = 'data/mnist' ;
opts.expDir = 'data/mnist-exp-1' ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.batchSize = 100 ;
opts.numEpochs = 100 ;
opts.continue = true ;
opts.useGpu = false ;
opts.learningRate = 0.001 ;
opts.learningRate = 0.0001 ;
opts.debug_weights = false;

run matlab/vl_setupnn ;

% --------------------------------------------------------------------
%                                                         Prepare data
% --------------------------------------------------------------------

if exist(opts.imdbPath)
  imdb = load(opts.imdbPath) ;
else
  imdb = getMnistImdb(opts) ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end

% Define network
f=1/100 ;
net.layers = {} ;
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', f*randn(5,5,1,20, 'single'), ...
                           'biases', zeros(1, 20, 'single'), ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'maxpool', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', f*randn(5,5,20,50, 'single'),...
                           'biases', zeros(1,50,'single'), ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'maxpool', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', f*randn(4,4,50,500, 'single'),...
                           'biases', zeros(1,500,'single'), ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'relu') ;
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', f*randn(1,1,500,10, 'single'),...
                           'biases', zeros(1,10,'single'), ...
                           'stride', 1, ...
                           'pad', 0) ;
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

imdb.images.data = bsxfun(@minus, imdb.images.data, mean(imdb.images.data,4)) ;
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
        - opts.learningRate/numel(batch) * res(l).dzdw{1} ;

      ly.biasesMomentum = 0.9 * ly.biasesMomentum ...
        - 0.0005 * opts.learningRate * ly.biases ...
        - opts.learningRate/numel(batch) * res(l).dzdw{2} ;

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
function imdb = getMnistImdb(opts)
% --------------------------------------------------------------------

files = {'train-images-idx3-ubyte', ...
         'train-labels-idx1-ubyte', ...
         't10k-images-idx3-ubyte', ...
         't10k-labels-idx1-ubyte'} ;

mkdir(opts.dataDir) ;
for i=1:4
  if ~exist(fullfile(opts.dataDir, files{i}), 'file')
    url = sprintf('http://yann.lecun.com/exdb/mnist/%s.gz',files{i}) ;
    fprintf('downloading %s\n', url) ;
    gunzip(url, opts.dataDir) ;
  end
end

f=fopen(fullfile(opts.dataDir, 'train-images-idx3-ubyte'),'r') ;
x1=fread(f,inf,'uint8');
fclose(f) ;
x1=permute(reshape(x1(17:end),28,28,60e3),[2 1 3]) ;

f=fopen(fullfile(opts.dataDir, 't10k-images-idx3-ubyte'),'r') ;
x2=fread(f,inf,'uint8');
fclose(f) ;
x2=permute(reshape(x2(17:end),28,28,10e3),[2 1 3]) ;

f=fopen(fullfile(opts.dataDir, 'train-labels-idx1-ubyte'),'r') ;
y1=fread(f,inf,'uint8');
fclose(f) ;
y1=double(y1(9:end)')+1 ;

f=fopen(fullfile(opts.dataDir, 't10k-labels-idx1-ubyte'),'r') ;
y2=fread(f,inf,'uint8');
fclose(f) ;
y2=double(y2(9:end)')+1 ;

imdb.images.data = single(reshape(cat(3, x1, x2),28,28,1,[])) ;
imdb.images.labels = cat(2, y1, y2) ;
imdb.images.set = [ones(1,numel(y1)) 3*ones(1,numel(y2))] ;
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.classes = arrayfun(@(x)sprintf('%d',x),0:9,'uniformoutput',false) ;
