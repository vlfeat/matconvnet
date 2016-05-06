function info = getVarsInfo(net)
%GETVARSINFO
%   Returns a struct with information on each var. Fields:
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
%     If type is 'input', this is empty (use name for struct NET.INPUTS).
%
%   `isDer`::
%     Whether the var is a derivative (all vars come in pairs, the main
%     var and its derivative)
%
%   `fanout`::
%     Indexes of layers (NET.FORWARD) that use this var in their inputs..

% Copyright (C) 2016 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).


  % number of vars, independent of net.vars (which is empty during eval).
  % just go through var references and find the highest index
  numVars = max([net.forward.outputVar, net.test.outputVar, ...
    net.params.var, structfun(@deal, net.inputs)']) ;
  if mod(numVars, 2) == 1  % odd-valued count; should be even to account for last var's derivative
    numVars = numVars + 1 ;
  end

  info = Net.initStruct(numVars, 'type', 'name', 'index', 'isDer', 'fanout') ;
  [info.isDer] = deal(false) ;

  % vars that correspond to inputs
  inputNames = fieldnames(net.inputs);
  for k = 1:numel(inputNames)
    var = net.inputs.(inputNames{k}) ;
    info(var).type = 'input' ;
    info(var).name = inputNames{k} ;
  end

  % vars that correspond to params
  var = [net.params.var] ;
  [info(var).type] = deal('param') ;
  [info(var).index] = deal(num2cell(1:numel(var))) ;
  [info(var).name] = deal(net.params.name) ;

  % vars that correspond to layer outputs
  var = [net.forward.outputVar] ;
  [info(var).type] = deal('layer') ;
  [info(var).index] = deal(num2cell(1:numel(var))) ;
  [info(var).name] = deal(net.forward.name) ;

  % vars that correspond to derivatives (every even-numbered var)
  info(2:2:end) = info(1:2:end-1) ;
  [info(2:2:end).isDer] = deal(true) ;

  % compute fanout (can probably optimize better)
  for k = 1:numel(net.forward)
    var = net.forward(k).inputVars ;
    for i = 1:numel(var)
      info(var(i)).fanout(end+1) = k ;
    end
  end
end
