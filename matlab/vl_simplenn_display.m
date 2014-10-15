function info = vl_simplenn_display(net, res)
% VL_SIMPLENN_DISPLAY  Simple CNN statistics
%    VL_SIMPLENN_DISPLAY(NET) prints statistics about the network NET.

% Copyright (C) 2014 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

fields={'layer', 'type', 'support', 'stride', 'pad', 'dim', 'fdim', 'field', 'mem'};
if nargin > 1
  fields = {fields{:}, 'xwhd', 'xmem', 'dxmem'} ;
end

do_print = (nargout == 0) ;

for w=fields
  switch char(w)
    case 'type', s = 'type' ;
    case 'stride', s = 'stride' ;
    case 'padding', s = 'pad' ;
    case 'field', s = 'rec. field' ;
    case 'dim', s = 'out dim' ;
    case 'fdim', s = 'filt dim' ;
    case 'mem', s = 'c/g net KB' ;
    case 'xwhd', s = 'x w/h/d' ;
    case 'xmem', s = 'c/g x MB' ;
    case 'dxmem', s = 'c/g dx MB' ;
    otherwise, s = char(w) ;
  end
  if do_print, fprintf('%10s',s) ; end
  for l=1:numel(net.layers)
    ly=net.layers{l} ;
    switch char(w)
      case 'layer', s=sprintf('%d', l) ;
      case 'type'
        switch ly.type
          case 'normalize', s='nrm';
          case 'pool', if strcmpi(ly.method,'avg'), s='apool'; else s='mpool'; end
          case 'conv', s='cnv' ;
          case 'softmax', s='sftm' ;
          case 'loss', s='lloss' ;
          case 'softmaxloss', 'sftml' ;
          otherwise s=ly.type ;
        end
      case 'support'
        switch ly.type
          case 'conv', support(1:2,l) = max([size(ly.filters,1) ; size(ly.filters,2)],1) ;
          case 'pool', support(1:2,l) = ly.pool(:) ;
          otherwise, support(1:2,l) = [1;1] ;
        end
        s=sprintf('%dx%d', support(1,l), support(2,l)) ;
      case 'fdim'
        switch ly.type
          case 'conv'
            filterDimension(l) = size(ly.filters,3) ;
            s=sprintf('%d', filterDimension(l)) ;
          otherwise
            filterDimension(l) = 0 ;
            s='n/a' ;
        end
      case 'stride'
        switch ly.type
          case {'conv', 'pool'}
            if numel(ly.stride) == 1
              stride(1:2,l) = ly.stride ;
            else
              stride(1:2,l) = ly.stride(:) ;
            end
          otherwise, stride(1:2,l)=1 ;
        end
        if all(stride(:,l)==stride(1,l))
          s=sprintf('%d', stride(1,l)) ;
        else
          s=sprintf('%dx%d', stride(1,l), stride(2,l)) ;
        end
      case 'pad'
        switch ly.type
          case {'conv', 'pool'}
            if numel(ly.pad) == 1
              pad(1:4,l) = ly.pad ;
            else
              pad(1:4,l) = ly.pad(:) ;
            end
          otherwise, pad(1:4,l)=0 ;
        end
        if all(pad(:,l)==pad(1,l))
          s=sprintf('%d', pad(1,l)) ;
        else
          s=sprintf('%d,%dx%d,%d', pad(1,l), pad(2,l), pad(3,l), pad(4,l)) ;
        end
      case 'field'
        for i=1:2
          field(i,l) = sum(cumprod([1 stride(i,1:l-1)]).*(support(i,1:l)-1))+1 ;
        end
        if all(field(:,l)==field(1,l))
          s=sprintf('%d', field(1,l)) ;
        else
          s=sprintf('%dx%d', field(1,l), field(2,l)) ;
        end
      case 'mem'
        [a,b] = xmem(ly) ;
        mem(1:2,l) = [a;b] ;
        s=sprintf('%.0f/%.0f', a/1024, b/1024) ;
      case 'dim'
        switch ly.type
          case 'conv', dimension(1,l) = size(ly.filters,4) ;
          otherwise
            if l > 1
              dimension(1,l) = dimension(1,l-1) ;
            end
        end
        s=sprintf('%d', dimension(1,l)) ;
      case 'xwhd'
        sz=size(res(l+1).x) ;
        s=sprintf('%dx%dx%d%d', sz(1), sz(2), sz(3), sz(4)')
      case 'xmem'
        [a,b]=xmem(res(l+1).x) ;
        rmem(1:2,l) = [a;b] ;
        s=sprintf('%.0f/%.0f', a/1024^2, b/1024^2) ;        
      case 'dxmem'
        [a,b]=xmem(res(l+1).dzdx) ;
        rmem(1:2,l) = [a;b] ;
        s=sprintf('%.0f/%.0f', a/1024^2, b/1024^2) ;
    end
    if do_print, fprintf('|%7s', s) ; end
  end
  if do_print, fprintf('|\n') ; end
end
if do_print
  [a,b] = xmem(net) ;
  fprintf('total network CPU/GPU memory: %.1f/%1.f MB\n', a/1024^2, b/1024^2) ;
  if nargin > 1
    [a,b] = xmem(res) ;
    fprintf('total result CPU/GPU memory: %.1f/%1.f MB\n', a/1024^2, b/1024^2) ;
  end
end

if nargout > 0
  info.support = support ;
  info.receptiveField = field ;
  info.stride = stride ;
  info.pad = pad ;
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


