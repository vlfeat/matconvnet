function imagenet()
run(fullfile(fileparts(mfilename('fullpath')), '../matlab/vl_setupnn.m')) ;
opts.expDir = 'data/imagenet-exp' ;
opts.dataDir = 'data/imagenet' ;
opts.numEpochs = 100 ;
opts.batchSize = 4 ;
opts.useGpu = false ;
opts.learningRate = 0.01 ;
opts.lite = true ;
mkdir(opts.expDir) ;

% -------------------------------------------------------------------------
%                                                    Network initialization
% -------------------------------------------------------------------------

net = initializeNetwork(opts);

for i=1:numel(net.layers)
  if ~strcmp(net.layers{i}.type,'conv'), continue; end
  net.layers{i}.filtersMomentum = zeros('like',net.layers{i}.filters) ; 
  net.layers{i}.biasesMomentum = zeros('like',net.layers{i}.biases) ;
end

if opts.useGpu
  net = vl_simplenn_move(net, 'gpu') ;
end

% -------------------------------------------------------------------------
%                                               Stochastic gradient descent
% -------------------------------------------------------------------------

if 0
  % generate fake database
  n = 100 ;
  v = 10 ;
  t = 10 ;
  imdb.images.id = ones(1, n+v+t) ;
  imdb.images.label = randi(1000, n+v+t) ;
  imdb.images.set = [ones(1,n) 2*ones(1,v) 3*ones(1,t)] ;
  imdb.images.name = cell(1, n+v+t) ;
else
  imdb = imdbFromImageNet(opts) ;
end

train = find(imdb.images.set==1) ;
val = find(imdb.images.set==2) ;

for epoch=1:opts.numEpochs
  modelPath = fullfile(opts.expDir, 'net-epoch-%d.mat') ;
  modelFigPath = fullfile(opts.expDir, 'net-train.pdf') ;
  if exist(sprintf(modelPath, epoch+1),'file'), continue ; end
  if exist(sprintf(modelPath, epoch), 'file')
    sprintf('resuming from epoch %d\n', epoch) ;
    load(sprintf(modelPath, epoch), 'net', 'valScores') ;
  end
  
  for t=1:opts.batchSize:numel(train)
    % get next image batch and labels
    batch = train(t:min(t+opts.batchSize-1, numel(train))) ;
    fprintf('training: epoch %02d: processing batch starting with image %d\n', epoch, batch(1)) ;
    [im, labels] = getBatch(opts, imdb, batch) ;
    
    % backprop
    net.layers{end}.class = labels ;
    res = vl_simplenn(net, im, 1) ;
    imdb.images.score(batch) = res(end-1).x(1,1,labels+(0:1000:1000*numel(batch)-1)) ;
    
    % gradient descent
    for l=1:numel(net.layers)
      ly = net.layers{l} ;
      if ~strcmp(ly.type, 'conv'), continue ; end
      
      ly.filtersMomentum = ...
        0.9 * ly.filtersMomentum - ...
        0.0005 * opts.learningRate * ly.filters - ...
        opts.learningRate * res(l).dzdw{1} ;
      ly.biasesMomentum = ...
        0.9 * ly.biasesMomentum - ...
        0.0005 * opts.learningRate * ly.biases - ...
        opts.learningRate * res(l).dzdw{2} ;
      
      ly.filters = ly.filters + ly.filtersMomentum ;
      ly.biases = ly.biases + ly.biasesMomentum ;
      net.layers{l} = ly ;
    end        
  end % next batch
  
  % evaluation on validation set
  for t=1:opts.batchSize:numel(val)
    batch = val(t:min(t+opts.batchSize-1, numel(val))) ;
    fprintf('validation: epoch %02d: processing batch starting with image %d\n', epoch, batch(1)) ;
    [im, labels] = getBatch(opts, imdb, batch) ;
    
    net.layers{end}.class = labels ;
    res = vl_simplenn(net, im) ;
    imdb.images.score(batch) = res(end-1).x(1,1,labels+(0:1000:1000*numel(batch)-1)) ;
  end
  
  % save
  trainScores(epoch) = mean(imdb.images.score(train)) ;
  valScores(epoch) = mean(imdb.images.score(val)) ;
  save(sprintf(modelPath,epoch), 'net', 'trainScores', 'valScores') ;
  
  figure(1) ; clf ;
  plot(trainScores, 'k--') ; hold on ;
  plot(valScores, 'b-') ; 
  xlabel('epoch') ; ylabel('energy') ; legend('train', 'val') ; grid on ;
  drawnow ;
  print(1, modelFigPath, '-dpdf') ;  
end

% -------------------------------------------------------------------------
function [im, labels] = getBatch(opts, imdb, batch)
% -------------------------------------------------------------------------
im = randn(227, 227, 3, numel(batch), 'single') ;
labels = imdb.images.label(batch) ;

% -------------------------------------------------------------------------
function net = initializeNetwork(opt)
% -------------------------------------------------------------------------

net.layers = {} ;

% Block 1
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', 0.01*randn(11, 11, 3, 96, 'single'), ...
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
                           'filters', 0.01*randn(5, 5, 48, 256, 'single'), ...
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
                           'filters', 0.01*randn(3,3,256,384,'single'), ...
                           'biases', ones(1,384,'single'), ...
                           'stride', 1, ...
                           'pad', 1) ;
net.layers{end+1} = struct('type', 'relu') ;

% Block 4
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', 0.01*randn(3,3,192,384,'single'), ...
                           'biases', ones(1,384,'single'), ...
                           'stride', 1, ...
                           'pad', 1) ;
net.layers{end+1} = struct('type', 'relu') ;

% Block 5
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', 0.01*randn(3,3,192,256,'single'), ...
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
                           'filters', 0.01*randn(6,6,256,4096,'single'),...
                           'biases', ones(1,4096,'single'), ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'relu') ;

% Block 7
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', 0.01*randn(1,1,4096,4096,'single'),...
                           'biases', ones(1,4096,'single'), ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'relu') ;

% Block 8
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', 0.01*randn(1,1,4096,1000,'single'), ...
                           'biases', zeros(1, 1000, 'single'), ...
                           'stride', 1, ...
                           'pad', 0) ;

% Block 9
net.layers{end+1} = struct('type', 'softmax') ;

% Block 10 loss
net.layers{end+1} = struct('type', 'loss') ;


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
