function imagenet()
run(fullfile(fileparts(mfilename('fullpath')), '../matlab/vl_setupnn.m')) ;
run ~/src/vlfeat/toolbox/vl_setup.m ;

opts.expDir = 'data/imagenet-exp' ;
opts.dataDir = 'data/imagenet' ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat') ;
opts.numEpochs = 100 ;
opts.batchSize = 4 ;
opts.useGpu = false ;
opts.learningRate = 0.001*5 ;
opts.continue = true ;
opts.lite = true ;
mkdir(opts.expDir) ;

% -------------------------------------------------------------------------
%                                                   Database initialization
% -------------------------------------------------------------------------

if exist(opts.imdbPath,'file')
  imdb = load(opts.imdbPath) ;
else
  imdb = imdbFromImageNet(opts) ;
  imdb.averageImage = [] ;
  train = find(imdb.images.set == 1) ;
  bs = 100 ;
  for t=1:bs:numel(train)
    batch = train(t:min(t+bs-1, numel(train))) ;
    fprintf('computing average image: processing batch starting with image %d\n', batch(1)) ;
    im{t} = mean(getBatch(opts, imdb, [], batch), 4) ;    
  end
  imdb.averageImage = mean(cat(4, im{:}),4) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
  clear im ;
end

% -------------------------------------------------------------------------
%                                                    Network initialization
% -------------------------------------------------------------------------

net = initializeNetwork(opts);
net.normalization.averageImage = imdb.averageImage ;

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

% -------------------------------------------------------------------------
%                                               Stochastic gradient descent
% -------------------------------------------------------------------------

train = find(imdb.images.set==1) ;
val = find(imdb.images.set==2) ;

rng(0) ;
train = train(randperm(numel(train))) ;
val = val(randperm(numel(val))) ;

for epoch=1:opts.numEpochs
  % fast-forward to where we stopped
  modelPath = fullfile(opts.expDir, 'net-epoch-%d.mat') ;
  modelFigPath = fullfile(opts.expDir, 'net-train.pdf') ;
  if opts.continue & exist(sprintf(modelPath, epoch),'file'), continue ; end
  if opts.continue & epoch > 1 & exist(sprintf(modelPath, epoch-1), 'file')
    fprintf('resuming from loading epoch %d\n', epoch-1) ;
    load(sprintf(modelPath, epoch-1), 'net', 'valScores') ;
  end
  
  objective = 0 ;
  for t=1:opts.batchSize:numel(train)
    % get next image batch and labels
    batch = train(t:min(t+opts.batchSize-1, numel(train))) ;
    batch_time = tic ;
    fprintf('training: epoch %02d: processing batch %3d of %3d ...', epoch, ...
      fix(t/opts.batchSize)+1, ceil(numel(train)/opts.batchSize)) ;
    [im, labels] = getBatch(opts, imdb, net, batch) ;    
        
    % backprop
    net.layers{end}.class = labels ;
    res = vl_simplenn(net, im, 1) ;
    objective = objective + double(gather(res(end).x)) ;
    
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
      
      bound(l) = max(ly.filters(:)) ;
    end
    figure(100) ;clf;   
    plot(bound);drawnow ;
    %if any(bound>15), keyboard; end
    
    % 13 is the last convolutional layer
    % 16 is the first fc layer
    
    
    batch_time = toc(batch_time) ;
    fprintf(' %.2f s (%.1f images/s)\n', batch_time, numel(batch)/ batch_time) ;
  end % next batch
  
  % evaluation on validation set
  valObjective = 0 ;
  for t=1:opts.batchSize:numel(val)
    batch = val(t:min(t+opts.batchSize-1, numel(val))) ;
    fprintf('validation: epoch %02d: processing batch %3d of %3d\n', epoch, ...
      fix(t/opts.batchSize)+1, ceil(numel(val)/opts.batchSize)) ;
    [im, labels] = getBatch(opts, imdb, [], batch) ;
    
    net.layers{end}.class = labels ;
    res = vl_simplenn(net, im) ;
    valObjective = valObjective + double(gather(res(end).x)) ;
  end
  
  % save
  trainScores(epoch) = objective / numel(train) ;
  valScores(epoch) = valObjective / numel(val) ;
  save(sprintf(modelPath,epoch), 'net', 'trainScores', 'valScores') ;
  
  figure(1) ; clf ;
  subplot(1,2,1) ;
  plot(1:epoch, trainScores, 'k--') ; hold on ;
  plot(1:epoch, valScores, 'b-') ; 
  xlabel('epoch') ; ylabel('energy') ; legend('train', 'val') ; grid on ;
  subplot(1,2,2) ;
  vl_imarraysc(net.layers{1}.filters) ;
  axis equal ;
  colormap gray ;
  drawnow ;  
  print(1, modelFigPath, '-dpdf') ;  
end

% -------------------------------------------------------------------------
function [im, labels] = getBatch(opts, imdb, net, batch)
% -------------------------------------------------------------------------
im = zeros(227, 227, 3, numel(batch), 'single') ;
for i=1:numel(batch)
  imt = imread(fullfile(imdb.imageDir, imdb.images.name{batch(i)})) ;
  if size(imt,3) == 1, imt = cat(3, imt, imt, imt) ; end
  w = size(imt,2) ;
  h = size(imt,1) ;
  % todo: replace with im2signel once scaling issues have been
  % understood
  if w > h
    imt = imresize(single(imt), [227, NaN]) ;
  else
    imt = imresize(single(imt), [NaN, 227]) ;
  end
  w = size(imt,2) ;
  h = size(imt,1) ;
  sx = (1:227) + round(w/2 - 227/2) ;
  sy = (1:227) + round(h/2 - 227/2) ;
  im(:,:,:,i) = max(0,min(1,imt(sy,sx,:))) ;
  
  % apply network normalization
  if ~isempty(net)
    im(:,:,:,i) = im(:,:,:,i) - net.normalization.averageImage ;
  end
end
if opts.useGpu
  im = gpuArray(im) ;
end
labels = imdb.images.label(batch) ;

% -------------------------------------------------------------------------
function net = initializeNetwork(opt)
% -------------------------------------------------------------------------

scal = 1 ;

net.layers = {} ;

% Block 1
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', 0.01/scal * randn(11, 11, 3, 96, 'single'), ...
                           'biases', ones(1, 96, 'single'), ...
                           'stride', 4, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'relu') ;
net.layers{end+1} = struct('type', 'pool', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'normalize', ...
                           'param', [5 1 0.0001/5 0.75], ...
                           'stride', 2, ...
                           'pad', 0) ;

% Block 2
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', 0.01/scal * randn(5, 5, 48, 256, 'single'), ...
                           'biases', ones(1, 256, 'single'), ...
                           'stride', 1, ...
                           'pad', 2) ;
net.layers{end+1} = struct('type', 'relu') ;
net.layers{end+1} = struct('type', 'pool', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'normalize', ...
                           'param', [5 1 0.0001/5 0.75], ...
                           'stride', 2, ...
                           'pad', 0) ;

% Block 3
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', 0.01/scal * randn(3,3,256,384,'single'), ...
                           'biases', ones(1,384,'single'), ...
                           'stride', 1, ...
                           'pad', 1) ;
net.layers{end+1} = struct('type', 'relu') ;

% Block 4
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', 0.01/scal * randn(3,3,192,384,'single'), ...
                           'biases', ones(1,384,'single'), ...
                           'stride', 1, ...
                           'pad', 1) ;
net.layers{end+1} = struct('type', 'relu') ;

% Block 5
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', 0.01/scal * randn(3,3,192,256,'single'), ...
                           'biases', ones(1,256,'single'), ...
                           'stride', 1, ...
                           'pad', 1) ;
net.layers{end+1} = struct('type', 'relu') ;
net.layers{end+1} = struct('type', 'pool', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', 0) ;

% Block 6
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', 0.01/scal * randn(6,6,256,4096,'single'),...
                           'biases', ones(1,4096,'single'), ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'relu') ;

% Block 7
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', 0.01/scal * randn(1,1,4096,4096,'single'),...
                           'biases', ones(1,4096,'single'), ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'relu') ;

% Block 8
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', 0.01/scal * randn(1,1,4096,1000,'single'), ...
                           'biases', zeros(1, 1000, 'single'), ...
                           'stride', 1, ...
                           'pad', 0) ;

% Block 9
if 0
  net.layers{end+1} = struct('type', 'softmax') ;
  net.layers{end+1} = struct('type', 'loss') ;
else
  net.layers{end+1} = struct('type', 'softmaxloss') ;
end

% -------------------------------------------------------------------------
function imdb = imdbFromImageNet(opts)
% -------------------------------------------------------------------------

meta = load(fullfile(opts.dataDir, 'ILSVRC2012_devkit_t12', 'data', 'meta.mat')) ;
names = textread(fullfile(opts.dataDir, 'imagesets', 'train.txt'), '%s')' ;
imageCats = regexp(names, '^[^/]+', 'match', 'once') ;
cats = {meta.synsets(1:1000).WNID} ;
descrs = {meta.synsets(1:1000).words} ;
[~,labels] = ismember(imageCats, cats) ;

imdb.images.id = 1:numel(names) ;
imdb.images.name = names ;
imdb.images.set = ones(1, numel(names)) ;
imdb.images.label = labels ;
imdb.cats.label = 1:numel(cats) ;
imdb.cats.name = cats ;
imdb.cats.description = descrs ;

names = textread(fullfile(opts.dataDir, 'imagesets', 'val.txt'), '%s')' ;
labels = textread(fullfile(opts.dataDir, 'ILSVRC2012_devkit_t12', 'data', 'ILSVRC2012_validation_ground_truth.txt'), '%d')' ;

imdb.images.id = horzcat(imdb.images.id, (1:numel(names)) + 1e7 - 1) ;
imdb.images.name = horzcat(imdb.images.name, names) ;
imdb.images.set = horzcat(imdb.images.set, 2*ones(1,numel(names))) ;
imdb.images.label = horzcat(imdb.images.label, labels) ;

names = textread(fullfile(opts.dataDir, 'imagesets', 'test.txt'), '%s')' ;
labels = zeros(1, numel(names)) ;

imdb.images.id = horzcat(imdb.images.id, (1:numel(names)) + 2e7 - 1) ;
imdb.images.name = horzcat(imdb.images.name, names) ;
imdb.images.set = horzcat(imdb.images.set, 3*ones(1,numel(names))) ;
imdb.images.label = horzcat(imdb.images.label, labels) ;

imdb.images.name = strcat(imdb.images.name, '.JPEG') ;
imdb.imageDir = fullfile(opts.dataDir, 'images') ;

if opts.lite
  for i=1:10
    sel = find(imdb.images.label == i) ;
    train = sel(imdb.images.set(sel) == 1) ;
    val = sel(imdb.images.set(sel) == 2) ;
    %test = sel(imdb.images.set(sel) == 3) ;
    train = train(1:10) ;
    val = val(1:3) ;
    %test = test(1:3) ;
    keep{i} = [train val] ;
  end
  test = find(imdb.images.set == 3) ;
  keep = sort(cat(2, keep{:}, test(1:30))) ;
  imdb.images.id = imdb.images.id(keep) ;
  imdb.images.name = imdb.images.name(keep) ;
  imdb.images.set = imdb.images.set(keep) ;
  imdb.images.label = imdb.images.label(keep) ;
  fid = fopen(fullfile(opts.expDir, 'image-list.txt'),'w') ;
  cellfun(@(x)fprintf(fid,'%s\n',x),imdb.images.name) ;
  fclose(fid) ;
end
