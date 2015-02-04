function [arcs, bufferNames] = vl_dagnn_getarcs(net, inputs)

assert(all(cellfun(@(a) isfield(a, 'name'), net.layers)));

if nargin == 2
  numInputs = numel(inputs);
  inputNames = {inputs.name};
else
  numInputs = 0;
  inputNames = {};
end

layers = net.layers;
bufferNames = [inputNames, ...
  cellfun(@(a) a.name, net.layers, 'UniformOutput', false)];
arcs = [];
for li = 1:numel(layers)
  linputs = layers{li}.inputs;
  [tfnd, pred] = ismember(linputs, bufferNames);
  if any(tfnd == 0)
    error('Inputs {%s} for layer %s not found', ...
      strjoin(linputs(~tfnd), ', '), layers{li}.name);
  end;
  pad = ones(1, numel(pred));
  arcs = [arcs [li*pad; pred(:)'; (li+numInputs)*pad]];
end
