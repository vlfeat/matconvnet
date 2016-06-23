function net = cnn_assafnet(varargin)
opts.k = [64 128 256 512];
opts.Nclass=10;%number of classses (CIFAR-10 / CIFAR-100)
opts.colorSpace = 'rgb';
opts.usePad = 1;
opts = vl_argparse(opts, varargin) ;

net = dagnn.DagNN; %network

%initial convolution
if strcmp(opts.colorSpace, 'gray')
    c = 1;
else
    c = 3;
end
id = 1;

opts.usePad = double(opts.usePad);

c = addConvBlock(net, id, [3 3], c, 16, 1); id = id + 1;

c = addConvBlock(net, id, [3 3], c, opts.k(1), opts.usePad); id = id + 1;
c = addConvBlock(net, id, [3 3], c, opts.k(1), opts.usePad); id = id + 1;

c = addConvBlock(net, id, [3 3], c, opts.k(2), opts.usePad); id = id + 1;
net.addLayer('pool1', dagnn.Pooling('poolSize',[1 1],'stride',[2 2]));
c = addConvBlock(net, id, [3 3], c, opts.k(2), opts.usePad); id = id + 1;

c = addConvBlock(net, id, [3 3], c, opts.k(3), opts.usePad); id = id + 1;
net.addLayer('pool2', dagnn.Pooling('poolSize',[1 1],'stride',[2 2]));
c = addConvBlock(net, id, [3 3], c, opts.k(3), opts.usePad); id = id + 1;

if opts.usePad
    net.addLayer('avg', dagnn.Pooling('poolSize',[8 8], 'method','avg'));
else
    c = addConvBlock(net, id, [3 3], c, opts.k(3), 0); id = id + 1;    
end

% net.addLayer('dropout', dagnn.DropOut());
net.addLayer('classify', dagnn.Conv('size', [1 1 opts.k(3) opts.Nclass], 'hasBias', true), [], 'prediction', {'classifyW', 'classifyB'});
net.addLayer('loss_softmaxlog', dagnn.Loss('loss', 'softmaxlog'), {'prediction', 'label'}, 'objective') ;
net.addLayer('loss_classerror', dagnn.Loss('loss', 'classerror'), {'prediction', 'label'}, 'error') ;

    
net.initParams();

%Meta parameters
net.meta.inputSize = [32 32 c] ;
net.meta.trainOpts.learningRate = [0.01*ones(1,2) 0.1*ones(1,80) 0.01*ones(1,40) 0.001*ones(1,40)] ;
net.meta.trainOpts.weightDecay = 0.0001 ;
net.meta.trainOpts.batchSize = 100 ;
net.meta.trainOpts.momentum = 0.9 ;
net.meta.trainOpts.numEpochs = numel(net.meta.trainOpts.learningRate) ;
end

function k_out = addConvBlock(net, id, filt_size, k_in, k_out, pad, stride)
if nargin < 6
    pad = 0;
end
if nargin < 7
    stride = 1;
end

ext = num2str(id);
net.addLayer(['conv' ext], dagnn.Conv('size', [filt_size k_in k_out], 'hasBias', false, 'stride',[stride stride], 'pad', [pad pad pad pad]), [], [], {['conv' ext 'W']});
f = net.getParamIndex(['conv' ext 'W']) ;
sc = sqrt(2/(filt_size(1)*filt_size(2)*k_out)) ; %improved Xavier
net.params(f).value = sc*randn([filt_size k_in k_out], 'single') ;
net.params(f).learningRate=1;
net.params(f).weightDecay=1;

net.addLayer(['bnrm' ext], dagnn.BatchNorm('numChannels', k_out), [], [], strcat(['bnrm' ext], {'M','B','X'}));
f = net.getParamIndex(['bnrm' ext 'M']);
net.params(f).value = ones(k_out, 1, 'single');
net.params(f).learningRate=2;
net.params(f).weightDecay=0;
f = net.getParamIndex(['bnrm' ext 'B']);
net.params(f).value = zeros(k_out, 1, 'single');
net.params(f).learningRate=1;
net.params(f).weightDecay=0;
f = net.getParamIndex(['bnrm' ext 'X']);
net.params(f).value = zeros(k_out, 2, 'single');
net.params(f).learningRate=0.5;
net.params(f).weightDecay=0;


net.addLayer(['relu' ext], dagnn.ReLU());
end
