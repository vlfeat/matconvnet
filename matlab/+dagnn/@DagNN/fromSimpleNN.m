function obj = fromSimpleNN(net, varargin)
% FROMSIMPLENN  Initialize a DagNN object from a SimpleNN network
%   FROMSIMPLENN(NET) initializes the DagNN object from the
%   specified CNN using the SimpleNN format.
%
%   SimpleNN objects are linear chains of computational layers. These
%   layers echange information through variables and parameters that
%   are not explicitly named.Hence, FROMSIMPLENN() uses a number of
%   rules to assign such names automatically:
%
%   * From the input to the output of the CNN, variables are called
%     `x0` (input of the first layer), `x1`, `x2`, .... In this
%     manner `xi` is the outut of the i-th layer.
%
%   * Any loss layer requires two inputs, the second being a label.
%     These are called `label` (for the first such layers), and then
%     `label2`, `label3`,... for any other similar layer.
%
%   Additinoally, the option `CanonicalNames` the function can change
%   the names of some variables to make them more convenient to
%   use. With this option turned on:
%
%   * The network input is called `input` instead of `x0`.
%
%   * The output of each SoftMax layer is called `prob` (or `prob2`,
%     ...).
%
%   * The output of each Loss layer is called `objective` (or `
%     objective2`, ...).
%
%   * The input of each SoftMax or Loss layer of type *softmax log
%     loss* is called `prediction` (or `prediction2`, ...). If a Loss
%     layer immediately follows a SoftMax layer, then the rule above
%     takes the precendence and the input name is not changed.
%
%   FROMSIMPLENN(___, 'OPT', VAL, ...) accepts the following options:
%
%   `CanonicalNames`:: false
%      If `true` use the rules above to assign more meaningful
%      names to some of the varibles.

% Copyright (C) 2015 Karel Lenc and Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

opts.canonicalNames = false ;
opts = vl_argparse(opts, varargin) ;

import dagnn.*

obj = DagNN() ;
net = vl_simplenn_move(net, 'cpu') ;

if isfield(net, 'normalization')
  obj.meta.normalization = net.normalization ;
end

for l = 1:numel(net.layers)
  inputs = {sprintf('x%d',l-1)} ;
  outputs = {sprintf('x%d',l)} ;

  params = struct(...
        'name', {}, ...
        'value', {}, ...
        'learningRate', [], ...
        'weightDecay', []) ;
  if isfield(net.layers{l}, 'name')
    name = net.layers{l}.name ;
  else
    name = sprintf('layer%d',l) ;
  end

  switch net.layers{l}.type
    case {'conv', 'convt'}
      if isfield(net.layers{l},'filters')
        sz = size(net.layers{l}.filters) ;
        hasBias = ~isempty(net.layers{l}.biases) ;
        params(1).name = sprintf('%sf',name) ;
        params(1).value = net.layers{l}.filters ;
        if hasBias
          params(2).name = sprintf('%sb',name) ;
          params(2).value = net.layers{l}.biases ;
        end
      else
        sz = size(net.layers{l}.weights{1}) ;
        hasBias = ~isempty(net.layers{l}.weights{2}) ;
        params(1).name = sprintf('%sf',name) ;
        params(1).value = net.layers{l}.weights{1} ;
        if hasBias
          params(2).name = sprintf('%sb',name) ;
          params(2).value = net.layers{l}.weights{2} ;
        end
      end
      if isfield(net.layers{l},'learningRate')
        params(1).learningRate = net.layers{l}.learningRate(1) ;
        if hasBias
          params(2).learningRate = net.layers{l}.learningRate(2) ;
        end
      end
      if isfield(net.layers{l},'weightDecay')
        params(1).weightDecay = net.layers{l}.weightDecay(1) ;
        if hasBias
          params(2).weightDecay = net.layers{l}.weightDecay(2) ;
        end
      end
      switch net.layers{l}.type
        case 'conv'
          block = Conv() ;
          block.size = sz ;
          if isfield(net.layers{l},'pad')
            block.pad = net.layers{l}.pad ;
          end
          if isfield(net.layers{l},'stride')
            block.stride = net.layers{l}.stride ;
          end
        case 'convt'
          block = ConvTranspose() ;
          block.size = sz ;
          if isfield(net.layers{l},'upsample')
            block.upsample = net.layers{l}.upsample ;
          end
          if isfield(net.layers{l},'crop')
            block.crop = net.layers{l}.crop ;
          end
          if isfield(net.layers{l},'numGroups')
            block.numGroups = net.layers{l}.numGroups ;
          end
      end
      block.hasBias = hasBias ;
    case 'pool'
      block = Pooling() ;
      if isfield(net.layers{l},'method')
        block.method = net.layers{l}.method ;
      end
      if isfield(net.layers{l},'pool')
        block.poolSize = net.layers{l}.pool ;
      end
      if isfield(net.layers{l},'pad')
        block.pad = net.layers{l}.pad ;
      end
      if isfield(net.layers{l},'stride')
        block.stride = net.layers{l}.stride ;
      end
    case {'normalize'}
      block = LRN() ;
      if isfield(net.layers{l},'param')
        block.param = net.layers{l}.param ;
      end
    case {'dropout'}
      block = DropOut() ;
      if isfield(net.layers{l},'rate')
        block.rate = net.layers{l}.rate ;
      end
      if isfield(net.layers{l},'frozen')
        block.frozen = net.layers{l}.frozen ;
      end
    case {'relu'}
      lopts = {} ;
      if isfield(net.layers{l}, 'leak'), lopts = {'leak', net.layers{l}} ; end
      block = ReLU('opts', lopts) ;
    case {'sigmoid'}
      block = Sigmoid() ;
    case {'softmax'}
      block = SoftMax() ;
    case {'softmaxloss'}
      block = Loss('loss', 'softmaxlog') ;
      % The loss has two inputs
      inputs{2} = getNewVarName(obj, 'label') ;
    case {'bnorm'}
      block = BatchNorm() ;
      if isfield(net.layers{l},'filters')
        params(1).name = sprintf('%sm',name) ;
        params(1).value = net.layers{l}.filters ;
        params(2).name = sprintf('%sb',name) ;
        params(2).value = net.layers{l}.biases ;
      else
        params(1).name = sprintf('%sm',name) ;
        params(1).value = net.layers{l}.weights{1} ;
        params(2).name = sprintf('%sb',name) ;
        params(2).value = net.layers{l}.weights{2} ;
      end
      if isfield(net.layers{l},'learningRate')
        params(1).learningRate = net.layers{l}.learningRate(1) ;
        params(2).learningRate = net.layers{l}.learningRate(2) ;
      end
      if isfield(net.layers{l},'weightDecay')
        params(1).weightDecay = net.layers{l}.weightDecay(1) ;
        params(2).weightDecay = net.layers{l}.weightDecay(2) ;
      end
    otherwise
      error([net.layers{l}.type ' is unsupported']) ;
  end

  obj.addLayer(...
    name, ...
    block, ...
    inputs, ...
    outputs, ...
    {params.name}) ;

  if ~isempty(params)
    findex = obj.getParamIndex(params(1).name) ;
    bindex = obj.getParamIndex(params(2).name) ;

    % if empty, keep default values
    if ~isempty(params(1).value)
      obj.params(findex).value = params(1).value ;
    end
    if ~isempty(params(2).value)
      obj.params(bindex).value = params(2).value ;
    end
    if ~isempty(params(1).learningRate)
      obj.params(findex).learningRate = params(1).learningRate ;
    end
    if ~isempty(params(2).learningRate)
      obj.params(bindex).learningRate = params(2).learningRate ;
    end
    if ~isempty(params(1).weightDecay)
      obj.params(findex).weightDecay = params(1).weightDecay ;
    end
    if ~isempty(params(2).weightDecay)
      obj.params(bindex).weightDecay = params(2).weightDecay ;
    end
  end
end

% --------------------------------------------------------------------
% Rename variables to canonical names
% --------------------------------------------------------------------

if opts.canonicalNames
  for l = 1:numel(obj.layers)
    if l == 1
      obj.renameVar(obj.layers(l).inputs{1}, 'input') ;
    end
    if isa(obj.layers(l).block, 'dagnn.SoftMax')
      obj.renameVar(obj.layers(l).outputs{1}, getNewVarName(obj, 'prob')) ;
      obj.renameVar(obj.layers(l).inputs{1}, getNewVarName(obj, 'prediction')) ;
    end
    if isa(obj.layers(l).block, 'dagnn.Loss')
      obj.renameVar(obj.layers(l).outputs{1}, 'objective') ;
      if isempty(regexp(obj.layers(l).inputs{1}, '^prob.*'))
        obj.renameVar(obj.layers(l).inputs{1}, ...
                      getNewVarName(obj, 'prediction')) ;
      end
    end
  end
end

% --------------------------------------------------------------------
function name = getNewVarName(obj, prefix)
% --------------------------------------------------------------------
t = 0 ;
name = prefix ;
while any(strcmp(name, {obj.vars.name}))
  t = t + 1 ;
  name = sprintf('%s%d', prefix, t) ;
end