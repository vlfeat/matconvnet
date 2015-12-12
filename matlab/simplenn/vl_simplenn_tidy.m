function tnet = vl_simplenn_tidy(net)
%VL_SIMPLENN_TIDY  Fix an incomplete or outdated SimpleNN network.
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

tnet = struct('layers', {{}}, 'meta', struct()) ;

% copy meta information in net.meta subfield
if isfield(net, 'classes')
  tnet.meta.classes = net.classes ;
end

if isfield(net, 'normalization')
  tnet.meta.normalization = net.normalization ;
end

if isfield(net, 'meta')
  tnet.meta = net.meta ;
end

% copy layers
for l = 1:numel(net.layers)
  defaults = {};
  layer = net.layers{l} ;

  % check weights format
  switch layer.type
    case {'conv', 'convt', 'bnorm'}
      if ~isfield(layer, 'weights')
        layer.weights = {...
          layer.filters, ...
          layer.biases} ;
        layer = rmfield(layer, 'filters') ;
        layer = rmfield(layer, 'biases') ;
      end
  end

  % check that weights inlcude moments in batch normalization
  if strcmp(layer.type, 'bnorm')
    if numel(layer.weights) < 3
      layer.weights{3} = ....
        zeros(numel(layer.weights{1}),2,'single') ;
    end
  end

  % fill in missing values
  switch layer.type
    case {'conv', 'pool'}
      defaults = [ defaults {...
        'pad', 0, ...
        'stride', 1, ...
        'opts', {}}] ;

    case 'convt'
      defaults = [ defaults {...
        'crop', 0, ...
        'upsample', 1, ...
        'numGroups', 1, ...
        'opts', {}}] ;

    case {'pool'}
      defaults = [ defaults {...
        'method', 'max', ...
        'pad', 0, ...
        'stride', 1, ...
        'opts', {}}] ;

    case 'relu'
      defaults = [ defaults {...
        'leak', 0}] ;

    case 'dropout'
      defaults = [ defaults {...
        'rate', 0.5}] ;

    case {'normalize', 'lrn'}
      defaults = [ defaults {...
        'param', [5 1 0.0001/5 0.75]}] ;
  end

  for i = 1:2:numel(defaults)
    if ~isfield(layer, defaults{i})
      layer.(defaults{i}) = defaults{i+1} ;
    end
  end

  % save back
  tnet.layers{l} = layer ;
end
