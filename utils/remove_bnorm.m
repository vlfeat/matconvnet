function net2 = remove_bnorm(net)

bnorm_layers = find(cellfun(@(l) strcmp(l.type, 'bnorm'), net.layers));
conv_layers = find(cellfun(@(l) strcmp(l.type, 'conv'), net.layers));
relu_layers = find(cellfun(@(l) strcmp(l.type, 'relu'), net.layers));

if ~all(ismember(bnorm_layers-1, conv_layers)) || ~all(ismember(bnorm_layers+1, relu_layers))
    error('Batch norm must be between conv & relu');
end

net2 = net;
for l = bnorm_layers
    if isa(net.layers{l}.weights{1},'gpuArray')
        error('Network must be in CPU mode (not GPU)');
    end
    % update weights
    sigma = reshape(net.layers{l}.weights{3}(:,2),1,1,1,[]);
    G = reshape(net.layers{l}.weights{1},1,1,1,[]);
    net2.layers{l-1}.weights{1} = bsxfun(@times, net.layers{l-1}.weights{1}, G./sigma);
    
    % update biases
    mu = reshape(net.layers{l}.weights{3}(:,1),1,[]);
    sigma = reshape(net.layers{l}.weights{3}(:,2),1,[]);
    G = reshape(net.layers{l}.weights{1},1,[]);
    B = reshape(net.layers{l}.weights{2},1,[]);
    net2.layers{l-1}.weights{2} = bsxfun(@plus, bsxfun(@times, bsxfun(@minus, net.layers{l-1}.weights{2}, mu), G./sigma), B);
end

net2.layers(bnorm_layers) = [];

if isfield(net,'meta') && isfield(net.meta, 'inputSize')
    
    netA = net;
    netB = net2;
    if ismember(net.layers{end}.type,{'pdist','softmaxloss','loss'})
        netA.layers(end) = [];
        netB.layers(end) = [];
    end
    
    data = rand([netA.meta.inputSize 128],'single');
    res = vl_simplenn(netA, data, [], [], 'mode', 'test');
    res(bnorm_layers) = [];
    res2 = vl_simplenn(netB, data, [], [], 'mode', 'test');
    assert(numel(res)==numel(res2))
    err = cell(size(res));
    for i = 1 : numel(res)
        err{i} = abs(res(i).x(:)-res2(i).x(:));
    end
    err = vertcat(err{:});    
    fprintf('Maximum error: %g\n' ,max(err));
    fprintf('Mean error: %g\n' ,mean(err));
end