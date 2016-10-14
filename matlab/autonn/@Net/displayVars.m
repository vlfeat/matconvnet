function displayVars(net, varargin)
%DISPLAYVARS
%   Simple table of information on each var, and corresponding derivative.
%
%   NET.DISPLAYVARS(VARS)
%   Uses the given variables list, rather than NET.VARS. This is useful
%   for debugging inside calls to NET.EVAL, where NET.VARS is empty and
%   a local variable VARS is used instead, for performance reasons.
%
%   NET.DISPLAYVARS(___, 'OPT', VAL, ...) accepts the following options:
%
%   `showRange`:: `true`
%      If set to true, shows columns with the minimum and maximum for each
%      variable.
%
%   `showLinks`:: `false`
%      If set to true, shows hyperlinks that print the syntax to access
%      the value of each variable (e.g. 'net.vars{INDEX}').


% Copyright (C) 2016 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).


if ~isempty(varargin) && iscell(varargin{1})
  vars = varargin{1} ;
  varargin(1) = [] ;
else
  vars = net.vars ;
end
opts.showRange = true ;
% opts.showLinks = usejava('desktop') ;
opts.showLinks = false ;  % disabled for now, since it causes misaligned headers
opts = vl_argparse(opts, varargin) ;

assert(~isempty(vars), 'NET.VARS is empty.') ;

info = net.getVarsInfo() ;
assert(numel(info) == numel(vars)) ;


% function of each layer, as a string. empty for params and inputs.
funcs = cell(1, numel(info)) ;
funcs([net.forward.outputVar]) = cellfun(@func2str, {net.forward.func}, 'UniformOutput', false) ;

% fill in remaining slots with 'param' or 'input', depending on the type
idx = ~strcmp({info.type}, 'layer') ;
funcs(idx) = {info(idx).type} ;

% print information for each variable
values = cell(numel(vars), 1) ;
flags = cell(numel(vars), 1) ;
mins = zeros(numel(vars), 1) ;
maxs = zeros(numel(vars), 1) ;
for i = 1:numel(vars)
  % size and underlying type (e.g. single 50x3x2)
  v = vars{i} ;
  if isa(v, 'gpuArray')
    str = classUnderlying(v) ;
  else
    str = class(v) ;
  end
  str = [str ' ' sprintf('%ix', size(v))] ;  %#ok<*AGROW>
  values{i} = str(1:end-1) ;  % remove extraneous 'x' at the end of the var size
  
  % flags (GPU, NaN, Inf)
  str = '' ;
  if isa(v, 'gpuArray')
    str = [str 'GPU '] ;
  end
  if any(isnan(v(:)))
    str = [str 'NaN '] ;
  end
  if any(isinf(v(:)))
    str = [str 'Inf '] ;
  end
  if isempty(str)
    flags{i} = ' ' ;  % no flags, must still have a space
  else
    flags{i} = str(1:end-1) ;  % delete extra space at the end
  end
  
  % values range
  if opts.showRange && ~isempty(v)
    mins(i) = gather(min(v(:))) ;
    maxs(i) = gather(max(v(:))) ;
  end
end

if opts.showRange
  mins = num2str(mins, 2) ;
  maxs = num2str(maxs, 2) ;
end


if opts.showLinks
  varname = inputname(1) ;
  link = @(i) sprintf('<a href="matlab: display(''%s.vars{%d}'')">%s</a>', ...
    varname, i, values{i}) ;
  values = arrayfun(@(i) link(i), 1:numel(values), 'UniformOutput', false) ;
end
idx = arrayfun(@(i) num2str(i), 1:2:numel(info)-1, 'UniformOutput', false) ;

% now print out the info as a table
str = repmat(' ', numel(info) / 2 + 1, 1) ;
str = [str, char('Idx', idx{:})] ;
str(:,end+1:end+2) = ' ' ;
str = [str, char('Type/function', funcs{1:2:end-1})] ;
str(:,end+1:end+2) = ' ' ;
str = [str, char('Name', info(1:2:end-1).name)] ;

names = {'Value', 'Derivative'} ;  % repeat same columns for value and der
for i = 1:2
  idx = i : 2 : numel(values) - 2 + i ;  % odd or even elements, respectively
  str(:,end+1:end+2) = ' ' ;
  str = [str, char(names{i}, values{idx})] ;
  str(:,end+1:end+2) = ' ' ;
  str = [str, char('Flags', flags{idx})] ;
  if opts.showRange
    str(:,end+1:end+2) = ' ' ;
    str = [str, char('Min', mins(idx,:))] ;
    str(:,end+1:end+2) = ' ' ;
    str = [str, char('Max', maxs(idx,:))] ;
  end
end

disp(str) ;
end
