function [arcs, bufferNames] = vl_dagnn_getarcs(net, inputs, doder)
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

if ~exist('doder', 'var'), doder = false; end;
numInputs = numel(inputs);
inputNames = {inputs.name};

has_names = cellfun(@(a) isfield(a, 'name') || isprop(a, 'name'), net.layers);
has_inputs = cellfun(@(a) isfield(a, 'inputs') || isprop(a, 'inputs'), net.layers);

if any(~has_inputs) || any(~has_names)
  error('Not a valid DAGNN network.');
end

layers = net.layers;
bufferNames = inputNames;
arcs_maps = {};

for li = 1:numel(layers)
  l = layers{li};
  linputs = l.inputs;
  l_doder = true;
  if isfield(l, 'dobp'), l_doder = l.dobp; end;
  
  % linputs=char     -> single input
  % linputs={char}   -> multiple inputs
  % linputs={{char}} -> weight sharing (apply layer multiple times)
  if ~iscell(linputs), linputs = {linputs};
  elseif ~iscell(linputs{1}), linputs = {linputs}; end;
 
  % Decide the output names
  if isfield(l, 'output')
    assert(iscell(l.output));
    bufferNames = [bufferNames l.output];
  elseif numel(linputs) == 1
    bufferNames = [bufferNames l.name];
  else
    error('Output names must be specified for layer %s.', l.name);
  end
  
  if numel(bufferNames) ~= numel(unique(bufferNames))
    error('Duplicate layer %s', bufferNames{end});
  end
  
  for wsi = 1:numel(linputs)
    [tfnd, input_ids] = ismember(linputs{wsi}, bufferNames);
    if any(tfnd == 0)
      error('Inputs {%s} for layer %s not found or are used before being computed.', ...
        strjoin(linputs{wsi}(~tfnd), ', '), l.name);
    end;
    arcs_maps{end+1} = struct(...
      'outputIdx', numel(arcs_maps)+1+numInputs, ...
      'layerIdx', li, ...
      'inputIdxs', input_ids(:)', ...
      'dzdxIdxs', input_ids(:)', ... % Basically needed just for the RELU hack :(
      'dobp', l_doder);
  end
end

arcs = struct('bufferNames', {bufferNames}, 'maps', cell2mat(arcs_maps));

% Create the counters for conserve memory
bufCounters = zeros(1, numel(arcs.bufferNames));
for mi = 1:numel(arcs.maps)
  iidxs = arcs.maps(mi).inputIdxs;
  bufCounters(iidxs) = bufCounters(iidxs) + 1;
  if doder && arcs.maps(mi).dobp
    bufCounters(iidxs) = bufCounters(iidxs) + 1;
  end
end
% Unused buffers are outputs
bufCounters(bufCounters == 0) = inf;
arcs.bufCounters = bufCounters;
