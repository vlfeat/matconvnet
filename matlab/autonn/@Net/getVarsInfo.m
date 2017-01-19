function info = getVarsInfo(net)
%GETVARSINFO
%   INFO = NET.GETVARSINFO()
%   Returns a struct INFO with information on each variable. Fields:
%
%   `type`::
%     Type of layer that outputs this var ('input', 'param', or 'layer').
%
%   `name`::
%     Name of the layer, input or param that outputs this var.
%
%   `index`::
%     Reference for the layer that outputs this var.
%     If type is 'layer', contains its index in NET.FORWARD.
%     If type is 'param', contains its index in NET.PARAMS.
%     If type is 'input', this is 0 (use name for struct NET.INPUTS).
%
%   `outputArgPos`::
%     For layers with multiple outputs, this is the index (argument
%     position) of the output. E.g. info(1).outputArgPos=2 means var 1 is
%     the second output of a layer.
%
%   `isDer`::
%     Whether the var is a derivative (all vars come in pairs, the main
%     var and its derivative)

% Copyright (C) 2016 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).


  % number of vars, independent of net.vars (which is empty during eval).
  % just go through var references and find the highest index
  numVars = max([net.forward.outputVar, net.params.var, structfun(@deal, net.inputs)']) ;
  if mod(numVars, 2) == 1  % odd-valued count; should be even to account for last var's derivative
    numVars = numVars + 1 ;
  end

  info = Net.initStruct(numVars, 'type', 'name', 'index', 'outputArgPos', 'isDer') ;
  [info.isDer] = deal(false) ;

  % vars that correspond to inputs
  inputNames = fieldnames(net.inputs);
  for k = 1:numel(inputNames)
    var = net.inputs.(inputNames{k}) ;
    info(var).type = 'input' ;
    info(var).index = 0 ;
    info(var).name = inputNames{k} ;
  end

  % vars that correspond to params
  var = [net.params.var] ;
  idx = num2cell(1:numel(var)) ;
  [info(var).type] = deal('param') ;
  [info(var).index] = deal(idx{:}) ;
  [info(var).name] = deal(net.params.name) ;

  % vars that correspond to layer outputs
  [info([net.forward.outputVar]).type] = deal('layer') ;
  
%   % optimized path for layers with a single output, which are more common
%   mask = (cellfun('length', {net.forward.outputVar}) == 1) ;
%   var = [net.forward(mask).outputVar] ;
%   [info(var).name] = deal(net.forward(mask).name) ;
%   idx = num2cell(find(mask)) ;
%   [info(var).index] = deal(idx{:}) ;
  
  for k = 1:numel(net.forward)
    var = net.forward(k).outputVar ;
    name = net.forward(k).name ;
    for v = 1:numel(var)
      info(var(v)).name = name ;
      info(var(v)).index = k ;
      info(var(v)).outputArgPos = v ;
    end
  end
  
  % vars that correspond to derivatives (every even-numbered var)
  info(2:2:end) = info(1:2:end-1) ;
  [info(2:2:end).isDer] = deal(true) ;
  
end
