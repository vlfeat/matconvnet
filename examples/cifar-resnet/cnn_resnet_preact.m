function net = cnn_resnet_preact(varargin)

%MatConvNet implementation of ReNets from:
%Identity Mappings in Deep Residual Networks (http://arxiv.org/abs/1603.05027)
%The implementation is based on the example from:
%https://github.com/KaimingHe/resnet-1k-layers

opts.networkType = 'dagnn' ;
opts.modelType='res';
opts.batchNormalization=1;
opts.depth=164;%164
opts.resConn=1;%residual connection (identity)
opts.Nclass=10;%number of classses (CIFAR-10 / CIFAR-100)
opts.resType = '131';
opts.colorSpace = 'rgb';
opts = vl_argparse(opts, varargin) ;

net= dagnn.DagNN; %network

switch opts.resType
    case '131'
        assert(~mod(opts.depth - 2,9))
        n = (opts.depth - 2) / 9; %stacks of residual units
    case '33'
        assert(~mod(opts.depth - 2,6))
        n = (opts.depth - 2) / 6; %stacks of residual units
    case 'HVHV'
        assert(~mod(opts.depth - 2,12))
        n = (opts.depth - 2) / 12; %stacks of residual units
end

resConn = opts.resConn; %residual connection
resType = opts.resType; %type of residual block

%initial convolution
if strcmp(opts.colorSpace,'gray')
    c = 1;
else
    c = 3;
end

%%
convBlock = dagnn.Conv('size', [3,3,c,16], 'pad', [1,1,1,1],'stride', [1,1], ...
    'hasBias', false);
net.addLayer('convInit',convBlock,{'input'},{'convInit'},{'convInit_filters'});
f = net.getParamIndex('convInit_filters') ;
sc = sqrt(2/(3*3*16)) ; %improved Xavier
net.params(f).value = sc*randn([3,3,c,16], 'single') ;
net.params(f).learningRate=1;
net.params(f).weightDecay=1;
layerInput = net.layers(end).name;

%Residual Model
switch lower(opts.modelType)
    case {'res'}%residual units
        filterDepths = [16, 64, 128, 256];
        stride = [1,2,2];
        
        %3-iterations(16->64, 64->128, 128->256)
        for i=1:numel(filterDepths)-1
            layerName=sprintf('bottleneck_%d_%d',filterDepths(i),filterDepths(i+1));
            [net,layerInput] = addStackedUnit(n, net, filterDepths(i), filterDepths(i+1), stride(i), layerName, layerInput, resConn, resType);
        end
    otherwise
        %code removed
end

%Final Batch Normalization
[net, layerInput] = addBnorm(net,layerInput,[0,0,0,filterDepths(end)],'bnormOut',[],[]);

%ReLU
x = net.getLayerIndex(layerInput);
inVar = net.layers(x).outputs;
net.addLayer('final_relu',  dagnn.ReLU(),{inVar{1}},{'final_relu'}, {});
layerInput='final_relu';

%Average Pool (input 8x8, output 1x1)
blockPool = dagnn.Pooling('method', 'avg', 'poolSize', [8 8], 'stride', 1, 'pad', [0,0,0,0]);
x = net.getLayerIndex(layerInput);
inVar = net.layers(x).outputs;
net.addLayer('avgPool', blockPool, {inVar{1}}, {'avgPool'}, {}) ;
layerInput = 'avgPool';

%Prediction
layerName='prediction';
iter=1;
elem=1;
[net, layerInput] = addConv(net, true, [1,1,filterDepths(end),opts.Nclass], [0,0,0,0], [1,1], layerName, layerInput, iter , elem);

%Modification from Andrea (similar to imagenet)
f = net.getParamIndex(net.layers(end).params(1)) ;
net.params(f).value = net.params(f).value /10;

prv_layerInput = layerInput;

%%
% net.addLayer('ds', dagnn.Pooling('poolSize', [1 1], 'stride', [2 2]), 'input', 'ds');
% 
% convBlock = dagnn.Conv('size', [3,3,c,16], 'pad', [1,1,1,1],'stride', [1,1], ...
%     'hasBias', false);
% net.addLayer('convInit2',convBlock,{'ds'},{'convInit2'},{'convInit_filters2'});
% f = net.getParamIndex('convInit_filters2') ;
% sc = sqrt(2/(3*3*16)) ; %improved Xavier
% net.params(f).value = sc*randn([3,3,c,16], 'single') ;
% net.params(f).learningRate=1;
% net.params(f).weightDecay=1;
% layerInput = net.layers(end).name;
% 
% %Residual Model
% switch lower(opts.modelType)
%     case {'res'}%residual units
%         filterDepths = [16, 64, 128, 256];
%         stride = [1,2,2];
%         
%         %3-iterations(16->64, 64->128, 128->256)
%         for i=1:numel(filterDepths)-1
%             layerName=sprintf('bottleneck2_%d_%d',filterDepths(i),filterDepths(i+1));
%             [net,layerInput] = addStackedUnit(n, net, filterDepths(i), filterDepths(i+1), stride(i), layerName, layerInput, resConn, resType);
%         end
%     otherwise
%         %code removed
% end
% 
% %Final Batch Normalization
% [net, layerInput] = addBnorm(net,layerInput,[0,0,0,filterDepths(end)],'bnormOut2',[],[]);
% 
% %ReLU
% x = net.getLayerIndex(layerInput);
% inVar = net.layers(x).outputs;
% net.addLayer('final_relu2',  dagnn.ReLU(),{inVar{1}},{'final_relu2'}, {});
% layerInput='final_relu2';
% 
% %Average Pool (input 8x8, output 1x1)
% blockPool = dagnn.Pooling('method', 'avg', 'poolSize', [4 4], 'stride', 1, 'pad', [0,0,0,0]);
% x = net.getLayerIndex(layerInput);
% inVar = net.layers(x).outputs;
% net.addLayer('avgPool2', blockPool, {inVar{1}}, {'avgPool2'}, {}) ;
% layerInput = 'avgPool2';
% 
% %Prediction
% layerName='prediction2';
% iter=1;
% elem=1;
% [net, layerInput] = addConv(net, true, [1,1,filterDepths(end),opts.Nclass], [0,0,0,0], [1,1], layerName, layerInput, iter , elem);
% 
% %Modification from Andrea (similar to imagenet)
% f = net.getParamIndex(net.layers(end).params(1)) ;
% net.params(f).value = net.params(f).value /10;
% 
% net.addLayer('sum', dagnn.Sum(), {layerInput prv_layerInput}, []);
% layerInput = 'sum';

%%
%Loss layer
x = net.getLayerIndex(layerInput);
inVar = net.layers(x).outputs;
net.addLayer('objective', dagnn.Loss('loss', 'softmaxlog'), ...
    {inVar{1},'label'}, 'objective') ;

%Error layer
net.addLayer('error', dagnn.Loss('loss', 'classerror'), ...
    {inVar{1},'label'}, 'error') ;

%Meta parameters
net.meta.inputSize = [32 32 3] ;
net.meta.trainOpts.learningRate = [0.01*ones(1,2) 0.1*ones(1,80) 0.01*ones(1,40) 0.001*ones(1,40)] ;
net.meta.trainOpts.weightDecay = 0.0001 ;
net.meta.trainOpts.batchSize = 128 ;
net.meta.trainOpts.momentum = 0.9 ;
net.meta.trainOpts.numEpochs = numel(net.meta.trainOpts.learningRate) ;

end

function [net,layerInput] = addResBlock(net, inDepth, outDepth, stride, layerName, layerInput, iter, resConn, resType)

bneckDepth = outDepth / 4;

switch resType
    case '131'
        %%% 1X1 %%%
        dims{1}    = [1,1,inDepth,bneckDepth];
        pad{1}     = [0,0,0,0];
        stride_{1} = [stride,stride];
        
        %%% 3X3 %%%
        dims{2}    = [3,3,bneckDepth,bneckDepth];
        pad{2}     = [1,1,1,1];
        stride_{2} = [1,1];
        
        %%% 1X1 %%%
        dims{3}    = [1,1,bneckDepth,outDepth];
        pad{3}     = [0,0,0,0];
        stride_{3} = [1,1];
    case '33'
        %%% 3X3 %%%
        dims{1}    = [3,3,inDepth,outDepth];
        pad{1}     = [1,1,1,1];
        stride_{1} = [stride,stride];
        
        %%% 3X3 %%%
        dims{2}    = [3,3,outDepth,outDepth];
        pad{2}     = [1,1,1,1];
        stride_{2} = [1,1];
    case 'HVHV'
        %%% 3X1 %%%
        dims{1}    = [3,1,inDepth,inDepth];
        pad{1}     = [1,1,0,0];
        stride_{1} = [stride,1];
        
        %%% 1X3 %%%
        dims{2}    = [1,3,inDepth,outDepth];
        pad{2}     = [0,0,1,1];
        stride_{2} = [1,stride];
        
        %%% 3X1 %%%
        dims{3}    = [3,1,outDepth,outDepth];
        pad{3}     = [1,1,0,0];
        stride_{3} = [1,1];
        
        %%% 1X3 %%%
        dims{3}    = [1,3,outDepth,outDepth];
        pad{3}     = [0,0,1,1];
        stride_{3} = [1,1];
        
    otherwise
        error('Unknown residual block type ''%s''', resType)
end

layerInputBr1=layerInput; %branch1 (identity)

for i=1:numel(dims);
    
    %Batch-normalization
    [net, layerInput] = addBnorm(net,layerInput,[0,0,0,dims{i}(3)],layerName,iter,i);
    
    %ReLU
    x = net.getLayerIndex(layerInput);
    inVar = net.layers(x).outputs;
    net.addLayer(sprintf('relu_%s_%d_%d',layerName,iter,i),  dagnn.ReLU(),...
        {inVar{1}},{sprintf('relu_%s_%d_%d',layerName,iter,i)}, {});
    layerInput=sprintf('relu_%s_%d_%d',layerName,iter,i);
    
    if inDepth ~= outDepth && i == 1 %increase depth
        %split here
        layerInputBr1=layerInput;
    end
    
    %Convolution
    [net, layerInput] = addConv(net, false, dims{i}, pad{i}, stride_{i}, layerName, layerInput, iter, i);
    
end

if resConn %residual connection (i.e summation)
    
    if inDepth ~= outDepth %increase depth
        %projection (match the input with the output depth)
        layerName_proj=sprintf('%s_projection',layerName);
        [net, layerInputBr1] = addConv(net, false, [1,1,inDepth,outDepth],[0,0,0,0], [stride,stride], layerName_proj, layerInputBr1, iter, 1);
    end
    
    %Summation
    x = net.getLayerIndex(layerInputBr1); %branch1
    inVar1 = net.layers(x).outputs;
    x = net.getLayerIndex(layerInput); %branch2
    inVar2 = net.layers(x).outputs;
    net.addLayer(sprintf('sum_%s_%d_%d',layerName,iter,i), dagnn.Sum(), ...
        {inVar1{1},inVar2{1}},sprintf('sum_%s_%d_%d',layerName,iter,i));
    layerInput = sprintf('sum_%s_%d_%d',layerName,iter,i);
end

end

function [net,layerInput] = addStackedUnit(N, net, inDepth, outDepth, stride, layerName, layerInput, resConn, resType)

iter=1;
[net,layerInput] = addResBlock(net, inDepth, outDepth, stride, layerName, layerInput,iter,resConn,resType);

stride_bneck=1;
for iter=2:N
    [net,layerInput] = addResBlock(net, outDepth, outDepth, stride_bneck, layerName, layerInput,iter, resConn, resType);
end

end

function [net, layerInput] = addBnorm(net,layerInput,dims,layerName,iterIdx,elemIdx)

x = net.getLayerIndex(layerInput);
inVar = net.layers(x).outputs;

params={sprintf('bn_%s_%d_%d_m',layerName,iterIdx,elemIdx),sprintf('bn_%s_%d_%d_b',layerName,iterIdx,elemIdx),sprintf('bn_%s_%d_%d_x',layerName,iterIdx,elemIdx)};
net.addLayer(sprintf('bn_%s_%d_%d',layerName,iterIdx,elemIdx), dagnn.BatchNorm(), {inVar{1}}, {sprintf('bn_%s_%d_%d',layerName,iterIdx,elemIdx)},params) ;
f = net.getParamIndex(sprintf('bn_%s_%d_%d_m',layerName,iterIdx,elemIdx));
net.params(f).value = ones(dims(4), 1, 'single');
net.params(f).learningRate=2;
net.params(f).weightDecay=0;
f = net.getParamIndex(sprintf('bn_%s_%d_%d_b',layerName,iterIdx,elemIdx));
net.params(f).value = zeros(dims(4), 1, 'single');
net.params(f).learningRate=1;
net.params(f).weightDecay=0;
f = net.getParamIndex(sprintf('bn_%s_%d_%d_x',layerName,iterIdx,elemIdx));
net.params(f).value = zeros(dims(4), 2, 'single');
net.params(f).learningRate=0.5;
net.params(f).weightDecay=0;

layerInput=sprintf('bn_%s_%d_%d',layerName,iterIdx,elemIdx);

end

function [net, layerInput] = addConv(net, has_bias, dims, pad, stride, layerName, layerInput, iterIdx, elemIdx)
% opts.cudnnWorkspaceLimit = 1024*1024*1024*1 ; % 1GB
% convOpts = {'CudnnWorkspaceLimit', opts.cudnnWorkspaceLimit} ;
convOpts = {};

x = net.getLayerIndex(layerInput);
inVar = net.layers(x).outputs;

convBlock = dagnn.Conv('size', dims, 'pad', pad,'stride', stride, ...
    'hasBias', has_bias, 'opts', convOpts);

if has_bias
    param_names = {sprintf('conv_%s_%d_%d_filters',layerName,iterIdx,elemIdx) ...
        sprintf('conv_%s_%d_%d_biases',layerName,iterIdx,elemIdx)};
else
    param_names = {sprintf('conv_%s_%d_%d_filters',layerName,iterIdx,elemIdx)};
end

net.addLayer(sprintf('conv_%s_%d_%d',layerName,iterIdx,elemIdx), ...
    convBlock, {inVar{1}}, {sprintf('conv_%s_%d_%d',layerName,iterIdx,elemIdx)},...
    param_names);

f = net.getParamIndex(sprintf('conv_%s_%d_%d_filters',layerName,iterIdx,elemIdx)) ;
sc = sqrt(2/(dims(1)*dims(2)*dims(4))) ; %improved Xavier
net.params(f).value = sc*randn(dims, 'single') ;
net.params(f).learningRate=1;
net.params(f).weightDecay=1;

layerInput = sprintf('conv_%s_%d_%d',layerName,iterIdx,elemIdx);

end