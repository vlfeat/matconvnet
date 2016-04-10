% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % net = toSimpleNN(dag, inputVar)                         %
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
% % dag - dagnn.DagNN object                                %
% % inputVar - [String] name of one and only input variable %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function net = toSimpleNN(dag, inputVar)
% Init simplenn struct
net = [];
net.layers = {};
net.meta = dag.meta;

% Init state
currentVars = { inputVar };
usedVars = {};

% Boolean array of visited layers
layerVisited = false(size(dag.layers));

splitSize = 1;
convSplitSeen = false;
while ~isempty(currentVars)
    % Find all layers that have current vars as input
    % And also other inputs are vars that have been seen before
    v = arrayfun(@(l) any(ismember(l.inputs, currentVars)) && ...
        all(ismember(l.inputs, currentVars) | ismember(l.inputs, usedVars)), ...
        dag.layers);
    
    % Make sure each layer is visited once or less
    if any(layerVisited(v))
        error('Layer was already visited');
    end    
    layerVisited(v) = true;
    
    layer = dag.layers(v);    
    usedVars = [usedVars currentVars];
    currentVars = [layer.outputs];
    
    if numel(layer) > 0
        if numel(layer) > 1
            if mod(numel(layer),splitSize)
                error('Split not allowed');
            end
            splitSize = 2;
            blocks = {layer.block};
            
            if ~all(cellfun(@(b) isa(b, class(blocks{1})), blocks))
                error('Blocks must be of the same type');
            end
            
            switch class(blocks{1})
                case 'dagnn.Conv'
                    if convSplitSeen
                        error('Not implemented yet'); % Verify filter groups support this
                    end
                    
                    % Verify blocks
                    convSplitSeen = true;
                    if ~all(cellfun(@(b) isequal(b.hasBias, blocks{1}.hasBias), blocks) & ...
                            cellfun(@(b) isequal(b.opts, blocks{1}.opts), blocks) & ...
                            cellfun(@(b) isequal(b.pad, blocks{1}.pad), blocks) & ...
                            cellfun(@(b) isequal(b.stride, blocks{1}.stride), blocks) & ...
                            cellfun(@(b) isequal(b.size(1:3), blocks{1}.size(1:3)), blocks))
                        error('Block configuration is not the same');
                    end
                    
                    % Create new block
                    block = dagnn.Conv();
                    block.load(blocks{1}.save);
                    block.size(4) = sum(cellfun(@(b) b.size(4),blocks));
                    
                    % Verify params
                    assert(all(arrayfun(@(l) dag.params(l.paramIndexes(1)).learningRate==dag.params(layer(1).paramIndexes(1)).learningRate,layer)));
                    assert(all(arrayfun(@(l) dag.params(l.paramIndexes(1)).weightDecay==dag.params(layer(1).paramIndexes(1)).weightDecay,layer)));
                    
                    % Merge params
                    params = dag.params(layer(1).paramIndexes);
                    params(1).value = arrayfun(@(l) dag.params(l.paramIndexes(1)).value, layer, 'UniformOutput', false);
                    params(2).value = arrayfun(@(l) dag.params(l.paramIndexes(2)).value, layer, 'UniformOutput', false);
                    params(1).value = cat(4, params(1).value{:});
                    params(2).value = cat(1, params(2).value{:});
                case 'dagnn.Sigmoid'
                    block = blocks{1};
                    params = [];
                case 'dagnn.ReLU'
                    if ~all(cellfun(@(b) isequal(b.leak, blocks{1}.leak), blocks))
                        error('Block configuration is not the same');
                    end
                    block = blocks{1};
                    params = [];
                otherwise
                    error('Unsupported layer type ''%s''', class(blocks{1}))
            end
        else
            % Only one layer at current stage
            % Take block and params
            block = layer.block;
            params = dag.params(layer.paramIndexes);
        end
        clear layer % Sanity
        
        % Add simplenn layer
        switch class(block)
            case 'dagnn.Conv'
                net.layers{end+1} = struct('type', 'conv', ...
                    'weights', {{params.value}}, ...
                    'learningRate', [params.learningRate], ...
                    'weightDecay', [params.weightDecay], ...
                    'stride', block.stride, ...
                    'pad', block.pad, ...
                    'opts', {block.opts}) ;
                if numel(net.layers{end}.weights)==1
                    net.layers{end}.weights{2} = [];
                end
            case 'dagnn.ReLU'
                net.layers{end+1} = struct('type', 'relu', ...
                    'leak', block.leak) ;
            case 'dagnn.BatchNorm'
                net.layers{end+1} = struct('type', 'bnorm', ...
                    'weights', {{params.value}}, ...
                    'learningRate', [params.learningRate], ...
                    'weightDecay', [params.weightDecay]) ;
            case 'dagnn.Pooling'
                net.layers{end+1} = struct('type', 'pool', ...
                    'method', block.method, ...
                    'pool', block.poolSize, ...
                    'stride', block.stride, ...
                    'pad', block.pad); 
            case 'dagnn.Sigmoid'
                net.layers{end+1} = struct('type', 'sigmoid') ;
            otherwise
                error('Unsupported layer type ''%s''', class(block))
        end
    end
end

net = vl_simplenn_tidy(net);
end

