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
    T = 10;
    maxErr = zeros(T,numel(net2.layers)+1);
    avgErr = zeros(T,numel(net2.layers)+1);
    for t = 1 : T
        data = rand(net.meta.inputSize,'single');
        res = vl_simplenn(net, data, [], [], 'mode', 'test');
        res(bnorm_layers) = [];
        res2 = vl_simplenn(net2, data, [], [], 'mode', 'test');
        assert(numel(res)==numel(res2))
        for i = 1 : numel(res)
            maxErr(t,i) = max(abs(res(i).x(:)-res2(i).x(:)));
            avgErr(t,i) = mean(abs(res(i).x(:)-res2(i).x(:)));
        end
    end
    fprintf('Maximum error: %g\n' ,max(maxErr(:)));
    fprintf('Mean error: %g\n' ,mean(avgErr(:)));
end