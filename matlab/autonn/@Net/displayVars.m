function displayVars(net, vars)
%DISPLAYVARS
%   Simple table of information on each var, and corresponding derivative.

% Copyright (C) 2016 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).


  if nargin < 2
    vars = net.vars ;
  end
  
  n = numel(vars) / 2 ;
  assert(n ~= 0, 'NET.VARS is empty.') ;
  type = cell(n, 1) ;
  names = cell(n, 1) ;
  funcs = cell(n, 1) ;

  % vars that correspond to inputs
  inputNames = fieldnames(net.inputs);
  for k = 1:numel(inputNames)
    idx = (net.inputs.(inputNames{k}) + 1) / 2 ;
    type{idx} = 'Input' ;
    names{idx} = inputNames{k} ;
  end

  % vars that correspond to params
  idx = ([net.params.var] + 1) / 2 ;
  [type{idx}] = deal('Param') ;
  names(idx) = {net.params.name} ;

  % vars that correspond to layer outputs
  idx = ([net.forward.outputVar] + 1) / 2 ;
  [type{idx}] = deal('Layer') ;
  names(idx) = {net.forward.name} ;
  funcs(idx) = cellfun(@func2str, {net.forward.func}, 'UniformOutput', false) ;

  % get Matlab to print the values nicely (e.g. "[50x1 double]")
  values = strsplit(evalc('disp(vars)'), '\n') ;
  values = cellfun(@strtrim, values, 'UniformOutput', false) ;

  % now print out the info as a table
  str = repmat(' ', n + 1, 1) ;
  str = [str, char('Type', type{:})] ;
  str(:,end+1:end+2) = ' ' ;
  str = [str, char('Function', funcs{:})] ;
  str(:,end+1:end+2) = ' ' ;
  str = [str, char('Name', names{:})] ;
  str(:,end+1:end+2) = ' ' ;
  str = [str, char('Value', values{1 : 2 : 2 * n})] ;
  str(:,end+1:end+2) = ' ' ;
  str = [str, char('Derivative', values{2 : 2 : 2 * n})] ;

  disp(str) ;
end