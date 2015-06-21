function [net, info] = cnn_googlenet_evaluate(varargin)
% CNN_GOOGLENET_EVALUATE   Evauate MatConvNet models on GoogleNet

run(fullfile(fileparts(mfilename('fullpath')), ...
  '..', 'matlab', 'vl_setupnn.m')) ;

opts.dataDir = fullfile('data', 'imagenet12') ;
opts.expDir = fullfile('data', 'imagenet-eval-googlenet') ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.modelPath = fullfile('data', 'models', 'imagenet-googlenet.mat') ;
opts.lite = false ;
opts.numFetchThreads = 8 ;
opts.train.batchSize = 20 ;
opts.train.numEpochs = 1 ;
opts.train.useGpu = true ;
opts.train.prefetch = false ;
opts.train.expDir = opts.expDir ;

opts = vl_argparse(opts, varargin) ;
display(opts);

% -------------------------------------------------------------------------
%                                                   Database initialization
% -------------------------------------------------------------------------

if exist(opts.imdbPath, 'file')
  fprintf('Loading imdb structure %s ...\n', opts.imdbPath);
  imdb = load(opts.imdbPath) ;
else
  fprintf('Creating the imdb structure...\n');
  imdb = cnn_imagenet_setup_data('dataDir', opts.dataDir, 'lite', opts.lite) ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end

% -------------------------------------------------------------------------
%                                                    Network initialization
% -------------------------------------------------------------------------
fprintf('Loading network %s ...\n', opts.modelPath);
net = load(opts.modelPath) ;
if isfield(net, 'net') ;
  net = net.net ;
  net.classes = imdb.classes ;
end
% Add multiclass-error layers
net.layers{end+1} = MultiClassErrorLayer([1 5], 'name', 'err1', ...
  'inputs', {'cls1_fc2', 'label'});
net.layers{end+1} = MultiClassErrorLayer([1 5], 'name', 'err2', ...
  'inputs', {'cls2_fc2', 'label'});
net.layers{end+1} = MultiClassErrorLayer([1 5], 'name', 'err3', ...
  'inputs', {'cls3_fc', 'label'});

% Define which outputs to collect and how to display them
outputDefs = {};
outputDefs{end+1} = struct(...
  'name', 'err1', 'layerName', 'err1', ...
  'printfun', @(err) fprintf('| E1 %.2f/%.2f ', err*100));
outputDefs{end+1} = struct(...
  'name', 'err2', 'layerName', 'err2', ...
  'printfun', @(err) fprintf('| E2 %.2f/%.2f ', err*100));
outputDefs{end+1} = struct(...
  'name', 'err3', 'layerName', 'err3', ...
  'printfun', @(err) fprintf('| E3 %.2f/%.2f ', err*100));
outputDefs = cell2mat(outputDefs);

% Synchronize label indexes between the model and the image database
imdb = cnn_imagenet_sync_labels(imdb, net);

% -------------------------------------------------------------------------
%                                               Stochastic gradient descent
% -------------------------------------------------------------------------
%net.normalization.keepAspect = 1;
fn = getBatchWrapper(net.normalization, opts.numFetchThreads) ;

[net, info] = cnn_dagtrain(net, imdb, fn, outputDefs, ...
  'conserveMemory', true, ...
  'getTrain', @(imdb, epoch) nan, ...
  'getVal', @(imdb, epoch) find(imdb.images.set==2), ...
  'plotResultsFun', @(info, opts) [], ...
  opts.train) ;

% -------------------------------------------------------------------------
function fn = getBatchWrapper(opts, numThreads)
% -------------------------------------------------------------------------
fn = @(imdb,batch) getBatch(imdb,batch,opts,numThreads) ;

% -------------------------------------------------------------------------
function inputs = getBatch(imdb, batch, opts, numThreads)
% -------------------------------------------------------------------------
images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
im = cnn_imagenet_get_batch(images, opts, ...
                            'numThreads', numThreads, ...
                            'prefetch', nargout == 0) ;
labels = imdb.images.label(batch) ;
inputs = struct('name', {'data', 'label'}, 'x', {im, labels});
