function net = cnn_imagenet_init_resnet(varargin)
% CNN_IMAGENET_INIT_RESNET  Initialize a standard CNN for ImageNet

opts.classNames = {} ;
opts.classDescriptions = {} ;
opts.averageImage = zeros(3,1) ;
opts.colorDeviation = zeros(3) ;
opts.cudnnWorkspaceLimit = 1024*1024*1204 ; % 1GB
opts = vl_argparse(opts, varargin) ;

net = dagnn.DagNN() ;

prevDepth = 3 ;
depth = 64 ;
inputVar = 'input' ;

function Conv(name, ksize, appendReLU, downsample, bias)
    if downsample
      stride = 2 ;
    else
      stride = 1 ;
    end
    if bias
      pars = {[name '_f'], [name '_b']} ;
    else
      pars = {[name '_f']} ;
    end
    net.addLayer([name  '_conv'], ...
      dagnn.Conv('size', [ksize ksize prevDepth depth], ...
      'stride', stride, ....
      'pad', (ksize-1)/2, ...
      'hasBias', bias, ...
      'opts', {'cudnnworkspacelimit', opts.cudnnWorkspaceLimit}), ...
      inputVar, ...
      [name '_conv'], ...
      pars) ;
    net.addLayer([name '_bn'], ...
      dagnn.BatchNorm('numChannels', depth), ...
      [name '_conv'], ...
      [name '_bn'], ...
      {[name '_bn_w'], [name '_bn_b'], [name '_bn_m']}) ;
    prevDepth = depth ;
    inputVar = [name '_bn'] ;
    if appendReLU
      net.addLayer([name '_relu'] , ...
                   dagnn.ReLU(), ...
                   inputVar, ...
                   [name '_relu']) ;
      inputVar = [name '_relu'] ;
    end
end

% -------------------------------------------------------------------------

% 7 x 7 conv + ReLU + down
Conv('conv1', 7, true, true, true) ;

% 3 x 3 maxpool
net.addLayer('conv1_pool' , ...
  dagnn.Pooling('poolSize', [3 3], 'stride', 2, 'pad', 1, 'method', 'max'), ...
  inputVar, ...
  'conv1') ;
inputVar = 'conv1' ;

% -------------------------------------------------------------------------

for s = 2:5

  switch s
    case 2, sectionLen = 3 ;
    case 3, sectionLen = 4 ; % 8 ;
    case 4, sectionLen = 6; % 23 ; % 36 ;
    case 5, sectionLen = 3 ;
  end
  for l = 1:sectionLen
    depth = 2^(s+4) ;
    sumVar = inputVar ;
    sumDepth = prevDepth ;
    name = sprintf('conv%d_%d', s, l)  ;

    % 1x1 conv + ReLU + downsample (if first in section and from conv3)
    Conv([name 'a'], 1, true, (s >= 3) & l == 1, false) ;
    Conv([name 'b'], 3, true, false, false) ;

    depth = 2^(s+6) ;
    Conv([name 'c'], 1, false, false, false) ;

    % addition
    if l == 1
      % When channging section, we need a 1x1 adapter for the sum
      net.addLayer([name '_adapt_conv'], ...
        dagnn.Conv('size', [1 1 sumDepth depth], ...
        'stride', 1 + (s >= 3), ...
        'hasBias', false, ...
        'opts', {'cudnnworkspacelimit', opts.cudnnWorkspaceLimit}), ...
        sumVar, ...
        [sumVar '_adapted'], ...
        [name 'adapt_f']) ;
      net.addLayer([name '_adapt_bn'], ...
        dagnn.BatchNorm('numChannels', depth), ...
        [sumVar '_adapted'], ...
        [sumVar '_adapted_bn'], ...
        {[name '_adapt_bn_w'], [name '_adapt_bn_b'], [name '_adapt_bn_m']}) ;
      sumVar = [sumVar '_adapted_bn'] ;
    end

    net.addLayer([name '_sum'] , ...
      dagnn.Sum(), ...
      {sumVar, inputVar}, ...
      [name '_sum']) ;
    inputVar = [name '_sum'] ;

    % sigma()
    net.addLayer([name '_relu'] , ...
      dagnn.ReLU(), ...
      inputVar, ...
      name) ;
    inputVar = name ;
  end
end

net.addLayer('prediction_avg' , ...
  dagnn.Pooling('poolSize', [7 7], 'method', 'avg'), ...
  inputVar, ...
  'prediction_avg') ;

net.addLayer('prediction' , ...
  dagnn.Conv('size', [1 1 2048 1000]), ...
  'prediction_avg', ...
  'prediction', ...
  {'prediction_f', 'prediction_b'}) ;

net.addLayer('loss', ...
  dagnn.Loss('loss', 'softmaxlog') ,...
  {'prediction', 'label'}, ...
  'objective') ;

net.addLayer('top1error', ...
             dagnn.Loss('loss', 'classerror'), ...
             {'prediction', 'label'}, ...
             'top1error') ;

net.addLayer('top5error', ...
             dagnn.Loss('loss', 'topkerror', 'opts', {'topK', 5}), ...
             {'prediction', 'label'}, ...
             'top5error') ;

% Meta parameters
net.meta.normalization.imageSize = [224 224 3] ;
net.meta.inputSize = [net.meta.normalization.imageSize, 32] ;
net.meta.normalization.cropSize = net.meta.normalization.imageSize(1) / 256 ;
net.meta.normalization.averageImage = opts.averageImage ;
net.meta.classes.name = opts.classNames ;
net.meta.classes.description = opts.classDescriptions ;
net.meta.augmentation.jitterLocation = true ;
net.meta.augmentation.jitterFlip = true ;
net.meta.augmentation.jitterBrightness = double(0.1 * opts.colorDeviation) ;
net.meta.augmentation.jitterAspect = [3/4, 4/3] ;
%net.meta.augmentation.jitterSaturation = 0.4 ;
%net.meta.augmentation.jitterContrast = 0.4 ;

net.meta.inputSize = {'input', [net.meta.normalization.imageSize 32]} ;

%lr = logspace(-1, -3, 60) ;
lr = [0.1 * ones(1,30), 0.01*ones(1,20), 0.001*ones(1,40)] ;
net.meta.trainOpts.learningRate = lr ;
net.meta.trainOpts.numEpochs = numel(lr) ;
net.meta.trainOpts.batchSize = 258 ;
net.meta.trainOpts.numSubBatches = 3 ;
net.meta.trainOpts.weightDecay = 0.0001 ;

% Init parameters randomly
net.initParams() ;

for l = 1:numel(net.layers)
  if isa(net.layers(l).block, 'dagnn.BatchNorm')
    k = net.getParamIndex(net.layers(l).params{3}) ;
    net.params(k).learningRate = 0.3 ;
  end
end

end