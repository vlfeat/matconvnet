function net = vl_simplenn_tidy(net)
%VL_SIMPLENN_TIDY  Fix an incomplete or outdated SimpleNN neural network.
%   NET = VL_SIMPLENN_TIDY(NET) takes the NET object and upgrades
%   it to the current version of MatConvNet. This is necessary in
%   order to allow MatConvNet to evolve, while maintaining the NET
%   objects clean.
%
%   The function is also generally useful to fill in missing default
%   values in NET.
%
%   See also: VL_SIMPLENN().

% Copyright (C) 2014-15 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

% move meta information in net.meta subfield
if isfield(net, 'classes')
  net.meta.classes = net.classes ;
  net = rmfield(net, 'classes') ;
end

if isfield(net, 'normalization')
  net.meta.normalization = net.normalization ;
  net = rmfield(net, 'normalization') ;
end

% rename filers, biases into weights{...}
for l = 1:numel(net.layers)
  switch net.layers{l}.type
    case {'conv', 'convt', 'bnorm'}
      if ~isfield(net.layers{l}, 'weights')
        net.layers{l}.weights = {...
          net.layers{l}.filters, ...
          net.layers{l}.biases} ;
        net.layers{l} = rmfield(net.layers{l}, 'filters') ;
        net.layers{l} = rmfield(net.layers{l}, 'biases') ;
      end
  end
end

% add moments to batch normalization if none is provided
for l = 1:numel(net.layers)
  if strcmp(net.layers{l}.type, 'bnorm')
    if numel(net.layers{l}.weights) < 3
      net.layers{l}.weights{3} = ....
        zeros(numel(net.layers{l}.weights{1}),2,'single') ;
    end
  end
end

% add default values for missing fields in layers
for l = 1:numel(net.layers)
  defaults = {} ;
  switch net.layers{l}.type
    case 'conv'
      defaults = {...
        'pad', 0, ...
        'stride', 1} ;

    case 'convt'
      defaults = {...
        'crop', 0, ...
        'upsample', 1, ...
        'numGroups', 1} ;

    case 'relu'
      defaults = {...
        'leak', 0} ;
  end

  for i = 1:2:numel(defaults)
    if ~isfield(net.layers{l}, defaults{i})
      net.layers{l}.(defaults{i}) = defaults{i+1} ;
    end
  end
end
