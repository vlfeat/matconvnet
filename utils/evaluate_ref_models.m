function evaluate_ref_models()
% Evaluate MatConvNet reference models to validate them

addpath(fullfile(fileparts(mfilename('fullpath')), '..','examples', 'imagenet')) ;

models = {...
  'caffe-ref', ...
  'caffe-alex', ...
  'vgg-s', ...
  'vgg-m', ...
  'vgg-f', ...
  'vgg-verydeep-19', ...
  'vgg-verydeep-16', ...
  'googlenet-dag'} ;

for i = 1:numel(models)
  opts.dataDir = fullfile('data', 'imagenet12-ram') ;
  opts.expDir = fullfile('data','models-eval', models{i}) ;
  opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
  opts.modelPath = fullfile('data', 'models', ...
    sprintf('imagenet-%s.mat', models{i})) ;
  opts.lite = false ;
  opts.numFetchThreads = 12 ;
  opts.train.batchSize = 128 ;
  opts.train.numEpochs = 1 ;
  opts.train.gpus = [1] ;
  opts.train.prefetch = true ;
  opts.train.expDir = opts.expDir ;

  resultPath = fullfile(opts.expDir, 'results.mat') ;
  if ~exist(resultPath)
    results = cnn_imagenet_evaluate(opts) ;
    save(fullfile(opts.expDir, 'results.mat'), 'results') ;
  end
end

fprintf('|%20s|%10s|%10s|%10s|\n', 'model', 'top-1 err.', 'top-5 err.', 'images/s') ;
fprintf('%s\n', repmat('-',1,20+10+10+10+5)) ;

for i = 1:numel(models)
  opts.expDir = fullfile('data', 'models-eval', models{i}) ;
  resultPath = fullfile(opts.expDir, 'results.mat') ;
  load(resultPath, 'results') ;

  if isfield(results.val, 'error')
    top5 = results.val.error(2,end) ;
    top1 = results.val.error(1,end) ;
    speed = results.val.speed(end) ;
  else
    top5 = results.val.top5err(1,end) ;
    top1 = results.val.top1err(1,end) ;
    speed = results.val.num / results.val.time ;
  end

  fprintf('|%20s|%10s|%10s|%10s|\n', ...
    models{i}, ...
    sprintf('%5.1f',top1*100), ...
    sprintf('%5.1f',top5*100), ...
    sprintf('%5.1f',speed)) ;
end
