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
  
  assert(~isempty(vars), 'NET.VARS is empty.') ;
  
  info = net.getVarsInfo() ;
  assert(numel(info) == numel(vars)) ;
  
  
  % function of each layer, as a string. empty for non-layers (e.g. params)
  funcs = cell(1, numel(info)) ;
  funcs([net.forward.outputVar]) = cellfun(@func2str, {net.forward.func}, 'UniformOutput', false) ;
  

  % get Matlab to print the values nicely (e.g. "[50x1 double]")
  values = strsplit(evalc('disp(vars)'), '\n') ;
  values = cellfun(@strtrim, values, 'UniformOutput', false) ;

  % now print out the info as a table
  str = repmat(' ', numel(info) / 2 + 1, 1) ;
  str = [str, char('Type', info(1:2:end-1).type)] ;
  str(:,end+1:end+2) = ' ' ;
  str = [str, char('Function', funcs{1:2:end-1})] ;
  str(:,end+1:end+2) = ' ' ;
  str = [str, char('Name', info(1:2:end-1).name)] ;
  str(:,end+1:end+2) = ' ' ;
  str = [str, char('Value', values{1:2:end-1})] ;
  str(:,end+1:end+2) = ' ' ;
  str = [str, char('Derivative', values{2:2:end})] ;

  disp(str) ;
end
