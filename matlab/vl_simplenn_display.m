function vl_simplenn_display(net)
% VL_SIMPLENN_DISPLAY  Simple CNN statistics
%    VL_SIMPLENN_DISPLAY(NET) prints statistics about the network NET.

% Copyright (C) 2014 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

for w={'layer', 'type', 'support', 'stride', 'padding', 'field', 'mem'}
  switch char(w)
    case 'type', s = 'type' ;
    case 'stride', s = 'stride' ;
    case 'padding', s = 'pad' ;
    case 'field', s = 'field' ;
    case 'mem', s = 'c//g mem' ;
    otherwise, s = char(w) ;
  end
  fprintf('%10s',s) ;
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
          case 'conv', support(1:2,l) = [size(ly.filters,1) ; size(ly.filters,2)] ;
          case 'pool', support(1:2,l) = ly.pool(:) ;
          otherwise, support(1:2,l) = [1;1] ;
        end
        s=sprintf('%dx%d', support(1,l), support(2,l)) ;
      case 'stride'
        switch ly.type
          case {'conv', 'pool'}
            if numel(ly.stride) == 1
              stride(1:2,l) = ly.stride ;
            else
              stride(1:2,l) = ly.stride(:) ;
            end
          otherwise, stride(:,l)=1 ;
        end
        s=sprintf('%dx%d', stride(1,l), stride(2,l)) ;
      case 'pad'
        switch ly.type
          case {'conv', 'pool'}
            if numel(ly.pad) == 1
              pad(1:2,l) = ly.pad ;
            else
              pad(1:2,l) = ly.pad(:) ;
            end
          otherwise, pad(:,l)=1 ;
        end
        s=sprintf('%dx%d', pad(1,l), pad(2,l)) ;
      case 'field'
        for i=1:2
          field(i,l) = sum(cumprod([1 stride(i,1:l-1)]).*(support(i,1:l)-1))+1 ;
        end
        s=sprintf('%dx%d', field(1,l), field(2,l)) ;
      case 'mem'
        [a,b] = xmem(ly) ;
        mem(1:2,l) = [a;b] ;
        s=sprintf('%.0f/%.0f', a/1024^2, b/1024^2) ;
    end
    fprintf('|%7s', s) ;    
  end
  fprintf('|\n') ;
end
fprintf('total CPU/GPU memory: %.1f/%1.f MB\n', sum(mem(1,:))/1024^2, sum(mem(2,:))/1024^2) ;


function [cpuMem,gpuMem]=xmem(s)
cpuMem = 0 ;
gpuMem = 0 ;
for f=fieldnames(s)'
  f = char(f) ;
  t=s.(f) ;
  if isstruct(t)
    [a,b] = xmem(t) ;
    cpuMem = cpuMem + a ;
    gpuMem = gpuMem + b ;
    m = m + xmem(t) ;
    continue ;
  end
  if isnumeric(t)
    if isa(t,'gpuArray')
      gpuMem = gpuMem + 4 * numel(t) ;
    else
      cpuMem = cpuMem + 4 * numel(t) ;
    end
  end
end

 

