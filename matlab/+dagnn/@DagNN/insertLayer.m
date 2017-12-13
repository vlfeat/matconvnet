function insertLayer(obj, name, block, inputs, outputs, params, varargin)
%INSERTLAYER  Inserts a layer to a DagNN
%   INSERTLAYER(NAME, LAYER, INPUTS, OUTPUTS, PARAMS) adds the
%   specified layer to the network. NAME is a string with the layer
%   name, used as a unique indentifier. BLOCK is the object
%   implementing the layer, which should be a subclass of the
%   Layer. INPUTS, OUTPUTS are cell arrays of variable names, and
%   PARAMS of parameter names.
%
%   See Also ADDLAYER().

index = find(strcmp(name, {obj.layers.name}), 1);
if ~isempty(index), error('There is already a layer with name ''%s''.', name), end

if nargin < 6, params = {}; end
if ischar(inputs), inputs = {inputs}; end
if ischar(outputs), outputs = {outputs}; end
if ischar(params), params = {params}; end
assert(numel(outputs) == 1);

% Replace var names
for i = 1:numel(inputs)
    for l = 1:numel(obj.layers)
        sel = strcmp(inputs{i}, obj.layers(l).inputs);
        [obj.layers(l).inputs{sel}] = deal(outputs{1});
    end
end

obj.addLayer(name, block, inputs, outputs, params, varargin{:});
