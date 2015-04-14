function info = vl_simplenn_display(net, varargin)
% VL_SIMPLENN_DISPLAY  Simple CNN statistics
%    VL_SIMPLENN_DISPLAY(NET) prints statistics about the network NET.
%
%    INFO=VL_SIMPLENN_DISPLAY(NET) returns instead a structure INFO
%    with several statistics for each layer of the network NET.
%
%    The function accepts the following options:
%
%    `inputSize`:: heuristically set
%       Specifies the size of the input tensor X that will be passed
%       to the network. This is used in order to estiamte the memory
%       required to process the network. If not specified,
%       VL_SIMPLENN_DISPLAY uses the value in
%       NET.NORMALIZATION.IMAGESIZE assuming a batch size of one
%       image, unless otherwise specified by the `batchSize` option.
%
%    `batchSize`:: 1
%       Specifies the number of data points in a batch in estimating
%       the memory consumption (see `inputSize`).

% Copyright (C) 2014-15 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

opts.inputSize = [] ;
opts.batchSize = 1 ;
opts = vl_argparse(opts, varargin) ;

fields={'layer', 'type', 'name', '-', ...
        'support', 'filtd', 'nfilt', 'stride', 'pad', 'rfield', '-', ...
        'osize', 'odepth', 'ocard', '-', ...
        'xmem', 'wmem'};

% get the support, stride, and padding of the operators
for l = 1:numel(net.layers)
  ly = net.layers{l} ;
  switch ly.type
    case 'conv'
      if isfield(ly, 'weights')
        info.support(1:2,l) = max([size(ly.weights{1},1) ; size(ly.weights{1},2)],1) ;
      else
        info.support(1:2,l) = max([size(ly.filters,1) ; size(ly.filters,2)],1) ;
      end
    case 'pool'
      info.support(1:2,l) = ly.pool(:) ;
    otherwise
      info.support(1:2,l) = [1;1] ;
  end
  if isfield(ly, 'stride')
    info.stride(1:2,l) = ly.stride(:) ;
  else
    info.stride(1:2,l) = 1 ;
  end
  if isfield(ly, 'pad')
    info.pad(1:4,l) = ly.pad(:) ;
  else
    info.pad(1:4,l) = 0 ;
  end
  for i=1:2
    info.receptiveField(i,l) = sum(cumprod([1 info.stride(i,1:l-1)]).*(info.support(i,1:l)-1))+1 ;
  end
end

% get the dimensions of the data
if ~isempty(opts.inputSize) ;
  info.size.x(1:4,1) = opts.inputSize(:) ;
elseif isfield(net, 'normalization') && isfield(net.normalization, 'imageSize')
  info.size.x(1:4,1) = [net.normalization.imageSize(:) ; opts.batchSize] ;
else
  info.size.x(1:4,1) = [NaN NaN NaN opts.batchSize] ;
end
for l = 1:numel(net.layers)
    ly = net.layers{l} ;
  if strcmp(ly.type, 'custom') && isfield(ly, 'getForwardSize')
    sz = ly.getForwardSize(ly, info.size.x(:,l)) ;
    info.size.x(:,l+1) = sz(:) ;
    continue ;
  end

  info.size.x(1, l+1) = floor((info.size.x(1,l) + ...
                               sum(info.pad(1:2,l)) - ...
                               info.support(1,l)) / info.stride(1,l)) + 1 ;
  info.size.x(2, l+1) = floor((info.size.x(2,l) + ...
                               sum(info.pad(3:4,l)) - ...
                               info.support(2,l)) / info.stride(2,l)) + 1 ;
  info.size.x(3, l+1) = info.size.x(3,l) ;
  info.size.x(4, l+1) = info.size.x(4,l) ;
  switch ly.type
    case 'conv'
      if isfield(ly, 'weights')
        f = ly.weights{1} ;
      else
        f = ly.filters ;
      end
      if size(f, 3) ~= 0
        info.size.x(3, l+1) = size(f,4) ;
      end
    case {'loss', 'softmaxloss'}
      info.size.x(3:4, l+1) = 1 ;
    case 'custom'
      info.size.x(3,l+1) = NaN ;
  end
end

if nargout > 0, return ; end

% print table
wmem = 0 ;
xmem = 0 ;
for w=fields
  switch char(w)
    case 'type', s = 'type' ;
    case 'stride', s = 'stride' ;
    case 'padding', s = 'pad' ;
    case 'rfield', s = 'rec field' ;
    case 'odepth', s = 'out depth' ;
    case 'osize', s = 'out size' ;
    case 'ocard', s = 'out card' ;
    case 'nfilt', s = 'num filt' ;
    case 'filtd', s = 'filt dim' ;
    case 'wmem', s = 'param mem' ;
    case 'xmem', s = 'data mem' ;
    case '-', s = '----------' ;
    otherwise, s = char(w) ;
  end
  fprintf('%10s',s) ;

  % do input pseudo-layer
  for l=0:numel(net.layers)
    switch char(w)
      case '-', s='-------' ;
      case 'layer', s=sprintf('%d', l) ;
      case 'osize', s=sprintf('%dx%d', info.size.x(1:2,l+1)) ;
      case 'odepth', s=sprintf('%d', info.size.x(3,l+1)) ;
      case 'ocard', s=sprintf('%d', info.size.x(4,l+1)) ;
      case 'xmem'
        a = prod(info.size.x(:,l+1)) * 4 ;
        s = pmem(a) ;
        xmem = xmem + a ;
      otherwise
        if l == 0
          if strcmp(char(w),'type'), s = 'input';
          else s = 'n/a' ; end
        else
          ly=net.layers{l} ;
          switch char(w)
            case 'name'
              if isfield(ly, 'name')
                s=ly.name(max(1,end-6):end) ;
              else
                s='' ;
              end
            case 'type'
              switch ly.type
                case 'normalize', s='norm';
                case 'pool', if strcmpi(ly.method,'avg'), s='apool'; else s='mpool'; end
                case 'softmax', s='softmx' ;
                case 'softmaxloss', s='softmxl' ;
                otherwise s=ly.type ;
              end

            case 'support'
              s=sprintf('%dx%d', info.support(1,l), info.support(2,l)) ;

            case 'nfilt'
              switch ly.type
                case 'conv'
                  if isfield(ly, 'weights'), a = size(ly.weights{1},4) ;
                  else, a = size(ly.filters,4) ; end
                  s=sprintf('%d',a) ;
                otherwise
                  s='n/a' ;
              end
            case 'filtd'
              switch ly.type
                case 'conv'
                  if isfield(ly, 'weights'), a = size(ly.weights{1},3) ;
                  else, a = size(ly.filters,3) ; end
                  s=sprintf('%d',a) ;
                otherwise
                  s='n/a' ;
              end

            case 'stride'
              if all(info.stride(:,l)==info.stride(1,l))
                s=sprintf('%d', info.stride(1,l)) ;
              else
                s=sprintf('%dx%d', info.stride) ;
              end
            case 'pad'
              if all(info.pad(:,l)==info.pad(1,l))
                s=sprintf('%d', info.pad(1,l)) ;
              else
                s=sprintf('%d,%dx%d,%d', info.pad) ;
              end
            case 'rfield'
              if all(info.receptiveField(:,l)==info.receptiveField(1,l))
                s=sprintf('%d', info.receptiveField(1,l)) ;
              else
                s=sprintf('%dx%d', info.receptiveField) ;
              end
            case 'wmem'
              a = 0 ;
              if isfield(ly, 'weights') ;
                for j=1:numel(ly.weights)
                  a = a + numel(ly.weights{j}) * 4 ;
                end
              end
              if isfield(ly, 'filters') ;
                a = a + numel(ly.filters) * 4 ;
              end
              if isfield(ly, 'biases') ;
                a = a + numel(ly.biases) * 4 ;
              end
              s = pmem(a) ;
              wmem = wmem + a ;
          end
        end
    end
    fprintf('|%7s', s) ;
  end
  fprintf('|\n') ;
end

fprintf('parameter memory: %s (%.2g parameters)\n', pmem(wmem), wmem/4) ;
fprintf('data memory: %s (batch size %d)\n', pmem(xmem), info.size.x(4,1)) ;

% -------------------------------------------------------------------------
function s= pmem(x)
% -------------------------------------------------------------------------
if isnan(x),       s = 'NaN' ;
elseif x < 1024^1, s = sprintf('%.0fB', x) ;
elseif x < 1024^2, s = sprintf('%.0fKB', x / 1024) ;
elseif x < 1024^3, s = sprintf('%.0fMB', x / 1024^2) ;
else               s = sprintf('%.0fGB', x / 1024^3) ;
end

% -------------------------------------------------------------------------
function [cpuMem,gpuMem] = xmem(s, cpuMem, gpuMem)
% -------------------------------------------------------------------------
if nargin <= 1
  cpuMem = 0 ;
  gpuMem = 0 ;
end
if isstruct(s)
  for f=fieldnames(s)'
    f = char(f) ;
    for i=1:numel(s)
      [cpuMem,gpuMem] = xmem(s(i).(f), cpuMem, gpuMem) ;
    end
  end
elseif iscell(s)
  for i=1:numel(s)
    [cpuMem,gpuMem] = xmem(s{i}, cpuMem, gpuMem) ;
  end
elseif isnumeric(s)
  if isa(s, 'single')
    mult = 4 ;
  else
    mult = 8 ;
  end
  if isa(s,'gpuArray')
    gpuMem = gpuMem + mult * numel(s) ;
  else
    cpuMem = cpuMem + mult * numel(s) ;
  end
end
