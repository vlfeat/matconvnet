function [arcs, bufferNames] = vl_dagnn_getarcs(net, inputs)
% VL_DAGNN_GETARCS Return the arcs in the DAG graph of a network
%   [ARCS, BUFFER_NAMES] = VL_DAGNN_GETARCS(NETWORK, INPUTS)
%   Computes the arcs in the NETWORK DAG graph. The graph is specified by
%   the fields 'name' and 'inputs' in the NETWORK.layers structures and the
%   INPUTS.name structure. Each column of ARCS specifies:
%
%     [LAYER_IDX; INPUT_BUFFER; OUTPUT_BUFFER]
%
%   Where the buffer names are given in the BUFFER_NAMES output.
%
%   See also VL_DAGNN, VL_CONVERT_SIMPLENN2DAGNN

% Copyright (C) 2014 Karel Lenc
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

if nargin == 2
  numInputs = numel(inputs);
  inputNames = {inputs.name};
else
  numInputs = 0;
  inputNames = {};
end

has_names = cellfun(@(a) isfield(a, 'name'), net.layers);
has_inputs = cellfun(@(a) isfield(a, 'inputs'), net.layers);

if any(~has_inputs) || ~any(has_names)
  error('Not a DAGNN network.');
end

layers = net.layers;
bufferNames = [inputNames, ...
  cellfun(@(a) a.name, net.layers, 'UniformOutput', false)];

arcs = [];
for li = 1:numel(layers)
  linputs = layers{li}.inputs;
  [tfnd, input_ids] = ismember(linputs, bufferNames);
  if any(tfnd == 0)
    error('Inputs {%s} for layer %s not found', ...
      strjoin(linputs(~tfnd), ', '), layers{li}.name);
  end;
  if any(input_ids >= (li+numInputs))
    error('Inputs {%s} for layer %s are used before being computed', ...
      strjoin(linputs(input_ids >= (li+numInputs)), ', '), layers{li}.name);
  end
  pad = ones(1, numel(input_ids));
  arcs = [arcs [li*pad; input_ids(:)'; (li+numInputs)*pad]];
end
