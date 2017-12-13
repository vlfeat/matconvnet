function [net, outputs, params] = addRNNLayer(net, name, block, inputs, outputs, params)
% Simple wrapper of dagnn.DagNN.addLayer()

% net.addLayer(name, block, inputs, outputs, params);
if iscell(name), name = name{1}; end
if ischar(inputs), inputs = {inputs}; end
if ischar(outputs), outputs = {outputs}; end
if ~exist('params', 'var'), params = {}; end

switch class(block)
    case 'dagnn.RNN2D'
        if isempty(params)
            params = {[name '_U'], [name '_W'], [name '_b']};
        elseif numel(params) ~= 3
            error('`%s` expects 3 parameters.', class(block));
        end
    case 'dagnn.LSTM2D'
        if strcmpi(block.style, 'normal')
            if isempty(params)
                params = {[name '_Wc'], [name '_Wi'], [name '_Wf'], [name '_Wo'], ...
                    [name '_Uc'], [name '_Ui'], [name '_Uf'], [name '_Uo'], ...
                    [name '_bc'], [name '_bi'], [name '_bf'], [name '_bo']};
            elseif numel(params) ~= 12
                error('`%s` expects 12 parameters.', class(block));
            end
            if numel(outputs) ~= 1
                error('`%s` expects 1 outputs.', class(block));
            end
        elseif strcmpi(block.style, 'coupled')
            if isempty(params)
                params = {[name '_Wc'], [name '_Wf'], [name '_Wo'], ...
                    [name '_Uc'], [name '_Uf'], [name '_Uo'], ...
                    [name '_bc'], [name '_bf'], [name '_bo']};
            elseif numel(params) ~= 9
                error('`%s` expects 9 parameters.', class(block));
            end
            if numel(outputs) ~= 1
                error('`%s` expects 1 outputs.', class(block));
            end
        elseif strcmpi(block.style, 'peephole')
            if isempty(params)
                params = {[name '_Wc'], [name '_Wi'], [name '_Wf'], [name '_Wo'], ...
                    [name '_Uc'], [name '_Ui'], [name '_Uf'], [name '_Uo'], ...
                    [name '_Vi'], [name '_Vf'], [name '_Vo'], ...
                    [name '_bc'], [name '_bi'], [name '_bf'], [name '_bo']};
            elseif numel(params) ~= 15
                error('`%s` expects 15 parameters.', class(block));
            end
            if numel(outputs) ~= 1
                error('`%s` expects 1 outputs.', class(block));
            end
        elseif strcmpi(block.style, 'gru')
            if isempty(params)
                params = {[name '_Wz'], [name '_Wr'], [name '_Wa'], ...
                    [name '_Uz'], [name '_Ur'], [name '_Ua'], ...
                    [name '_bz'], [name '_br'], [name '_ba']};
            elseif numel(params) ~= 9
                error('`%s` expects 9 parameters.', class(block));
            end
            if numel(outputs) ~= 1
                error('`%s` expects 1 outputs.', class(block));
            end
        end
end
net.addLayer(name, block, inputs, outputs, params);
