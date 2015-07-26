function addLayer(obj, name, block, inputs, outputs, params)
% OBJ.ADDLAYER(NAME, LAYER, INPUTS, OUTPUTS, PARAMS) adds the
% specified layer to the network. NAME is a string with the layer
% name, used as a unique indentifier. BLOCK is the object implementing
% the layer, which should be a subclass of the Layer. INPUTS, OUTPUTS
% are cell arrays of variable names, and PARAMS of parameter names.

f = find(strcmp(name, {obj.layers.name})) ;
if ~isempty(f), error('There is already a layer with name ''%s''.', name), end
f = numel(obj.layers) + 1 ;

if nargin < 6, params = {} ; end
if isstr(inputs), inputs = {inputs} ; end
if isstr(outputs), outputs = {outputs} ; end
if isstr(params), params = {params} ; end

obj.layers(f) = struct(...
  'name', {name}, ...
  'inputs', {inputs}, ...
  'outputs', {outputs}, ...
  'params', {params}, ...
  'inputIndexes', {[]}, ...
  'outputIndexes', {[]}, ...
  'paramIndexes', {[]}, ...
  'block', {block}) ;

block.net = obj ;

for input = inputs
  v = obj.addVar(char(input)) ;
  obj.vars(v).fanout = obj.vars(v).fanout + 1 ;
end

for output = outputs
  v = obj.addVar(char(output)) ;
  obj.vars(v).fanin = obj.vars(v).fanin + 1 ;
end

for param = params
  p = obj.addParam(char(param)) ;
  obj.params(p).fanout = obj.params(p).fanout + 1 ;
end

obj.rebuild() ;
