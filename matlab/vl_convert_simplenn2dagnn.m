function [ net ] = vl_convert_simplenn2dagnn( net )
% VL_CONVERT_SIMPLENN2DAGNN Convert SIMPLENN net to DAGNN net
%   DAG_NET = VL_CONVERT_SIMPLENN2DAGNN(CHAIN_NET) Converts a network with
%   a chain topology (e.g. one evaluated with vl_simplenn) to a network
%   with DAG topology.
%
%   The output network DAG_NET will expect to be passed to vl_dagnn with an
%   input with name 'data' and 'label' in case some loss layer is present.
%
%   In case the network layer names are not specified, layer names are
%   generated as 'l<layer_idx>'.
%
%   See also: vl_dagnn

% Copyright (C) 2014 Karel Lenc
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

has_names = cellfun(@(a) isfield(a, 'name'), net.layers);
has_inputs = cellfun(@(a) isfield(a, 'inputs'), net.layers);

if any(has_inputs), error('Not a simplenn network.'); end

if ~all(has_names)
  for li = 1:numel(net.layers)
    net.layers{li}.name = sprintf('l%d', li);
  end
end

for li = 1:numel(net.layers)
  if li == 1
    net.layers{li}.inputs = {'data'};
  else
    net.layers{li}.inputs = {net.layers{li-1}.name};
    if ismember(net.layers{li}.type, {'softmaxloss', 'loss'})
      net.layers{li}.inputs(end+1) = {'label'};
    end
  end
end

end

