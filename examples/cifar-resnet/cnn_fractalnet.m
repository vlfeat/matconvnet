function net = cnn_fractalnet(varargin)
opts.depth=16;%164
opts.k = [16 64 128 256];
opts.Nclass=10;%number of classses (CIFAR-10 / CIFAR-100)
opts.colorSpace = 'rgb';
opts = vl_argparse(opts, varargin) ;

net = dagnn.DagNN; %network

n_stacks = numel(opts.k);
assert(~mod(opts.depth, n_stacks))
n = (opts.depth) / n_stacks; %stacks of residual units
assert(log2(n)==floor(log2(n)));
n = log2(n)+1;

%initial convolution
if strcmp(opts.colorSpace, 'gray')
    c = 1;
else
    c = 3;
end

addStack(net, 1, c, opts.k(1), n, 'input');     
for s = 2 : n_stacks
    net.addLayer(sprintf('pool%d', s-1), dagnn.Pooling('poolSize', [3 3], 'stride', [2 2], 'pad', [1 1 1 1]));
    input_name = net.vars(net.layers(end).outputIndexes).name;
    addStack(net, s, opts.k(s-1), opts.k(s), n, input_name);    
end

if n_stacks ~= 5
    ps = 32/2^(n_stacks-1);
    net.addLayer('avg', dagnn.Pooling('method', 'avg', 'poolSize', [ps ps]));
end
% net.addLayer('dropout', dagnn.DropOut());
net.addLayer('classify', dagnn.Conv('size', [1 1 opts.k(end) opts.Nclass], 'hasBias', true), [], 'prediction', {'classifyW', 'classifyB'});
net.addLayer('loss_softmaxlog', dagnn.Loss('loss', 'softmaxlog'), {'prediction', 'label'}, 'objective') ;
net.addLayer('loss_classerror', dagnn.Loss('loss', 'classerror'), {'prediction', 'label'}, 'error') ;
net.initParams();

%Meta parameters
net.meta.inputSize = [32 32 3] ;
net.meta.trainOpts.learningRate = [0.01*ones(1,2) 0.1*ones(1,80) 0.01*ones(1,40) 0.001*ones(1,40)] ;
net.meta.trainOpts.weightDecay = 0.0001 ;
net.meta.trainOpts.batchSize = 128 ;
net.meta.trainOpts.momentum = 0.9 ;
net.meta.trainOpts.numEpochs = numel(net.meta.trainOpts.learningRate) ;
end

function output_name = addStack(net, stack_id, k_in, k_out, n, input_name)
output_name = addFractal(net, stack_id, k_in, k_out, n, input_name, 1);
end

function output_name = addFractal(net, stack_id, k_in, k_out, n, input_name, i)

if i < 2^n
    b = dec2bin(i,n);
    msb = find(b=='1',1,'first');
    b(msb) = '0';
    depth_id = bin2dec(b)+1;
    column_id = n-msb+1;
    
    ext = sprintf('_%d_%d_%d', stack_id, column_id, depth_id);
    net.addLayer(['conv' ext], dagnn.Conv('size', [3 3 k_in k_out], 'hasBias', false, 'pad', [1 1 1 1]), input_name, [], {['conv' ext 'W']});
    net.addLayer(['bnrm' ext], dagnn.BatchNorm('numChannels', k_out), [], [], strcat(['bnrm' ext], {'M','B','X'}));
    net.addLayer(['relu' ext], dagnn.ReLU());
    tmp_name = net.vars(net.layers(end).outputIndexes).name;
    
    output_name = addFractal(net, stack_id, k_in, k_out, n, input_name, 2*i);
    if isempty(output_name), 
        output_name = tmp_name;
    else
        output_name = addFractal(net, stack_id, k_out, k_out, n, output_name, 2*i+1);    
        
        src = find(arrayfun(@(l) strcmp(l.outputs{1}, output_name), net.layers));
        
        if isa(net.layers(src).block, 'dagnn.Join') 
            % Remove layer and replace with joined join.            
            assert(all(~arrayfun(@(l) any(strcmp(l.inputs, output_name)), net.layers))); % Sanity: make sure layer is not used
            inp = net.layers(src).inputs;
            net.removeLayer(net.layers(src).name)
            net.addLayer(['join' ext], dagnn.Join(), {tmp_name inp{:}});
        else        
            net.addLayer(['join' ext], dagnn.Join(), {tmp_name output_name});
        end
        
        output_name = net.vars(net.layers(end).outputIndexes).name;
    end
else
    output_name = {};
end

end

function addConvBlock(net, layer_name, filt_size, pad, hasBias)
net.add(layer_name, dagnn.Conv('size', filt_size, 'pad', [pad pad pad pad], 'hasBias', hasBias), [], [], strcat(layer_name, {'W','B'}));
end
