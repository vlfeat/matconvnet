function vl_simplenn_display(net)
% VL_SIMPLENN_DISPLAY  Simple CNN statistics
%    VL_SIMPLENN_DISPLAY(NET) prints statistics about the network NET.

% Copyright (C) 2014 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

for w={'layer', 'type', 'rho', 'delta', 'rc'}
  switch char(w)
    case 'rho', s = '$\rho_l$' ;
    case 'delta', s = '$\delta_l$' ;
    case 'rc', s = 'r.f.' ;
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
          case 'pool', s='mp' ;
          case 'conv', s='cnv' ;
          case 'softmax', s='sftm' ;
          case 'fully', s='full' ;
          otherwise s=ly.type ;
        end
      case 'rho'
        switch ly.type
          case 'conv', rho(l)=size(ly.filters,1) ;
          case 'pool', rho(l)=ly.pool(1) ;
          otherwise, rho(l)=1;
        end
        s=sprintf('%d', rho(l)) ;
      case 'delta'
        switch ly.type
          case 'conv', delta(l)=ly.stride;
          case 'pool', delta(l)=ly.stride;
          otherwise, delta(l)=1;
        end
        s=sprintf('%d', delta(l)) ;
      case 'rc'
        rc(l)=sum(cumprod([1 delta(1:l-1)]).*(rho(1:l)-1))+1 ;
        s=sprintf('%d', rc(l)) ;
    end
    fprintf('& %5s', s) ;    
  end
  fprintf('\\\\\n') ;
end