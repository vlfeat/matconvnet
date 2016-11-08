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
%   `showLinks`:: `true`
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
opts.showLinks = usejava('desktop') ;
opts = vl_argparse(opts, varargin) ;

assert(~isempty(vars), 'NET.VARS is empty.') ;

info = net.getVarsInfo() ;
assert(numel(info) == numel(vars)) ;


% print information for each variable
funcs = cell(numel(vars), 1) ;
values = cell(numel(vars), 1) ;
flags = cell(numel(vars), 1) ;
mins = zeros(numel(vars), 1) ;
maxs = zeros(numel(vars), 1) ;
validRange = true(numel(vars), 1) ;
for i = 1:numel(vars)
  % function of each layer, as a string
  if strcmp(info(i).type, 'layer')
    funcs{i} = func2str(net.forward(info(i).index).func) ;
    if info(i).outputArgPos > 1
      funcs{i} = sprintf('%s (output #%i)', funcs{i}, info(i).outputArgPos) ;
    end
  else
    % other var types like 'param' or 'input' will be displayed as such
    funcs{i} = info(i).type ;
  end
  
  % size and underlying type (e.g. single 50x3x2)
  v = gather(vars{i}) ;
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
  
  % min and max
  if opts.showRange
    if isnumeric(v) && ~isempty(v)
      mins(i) = gather(min(v(:))) ;
      maxs(i) = gather(max(v(:))) ;
    else
      validRange(i) = false ;
    end
  end
end

if opts.showRange  % convert to string, filling invalid values with spaces
  minStr(validRange,:) = num2str(mins(validRange), '%.2g') ;
  maxStr(validRange,:) = num2str(maxs(validRange), '%.2g') ;
  minStr(~validRange,:) = ' ' ;
  maxStr(~validRange,:) = ' ' ;
end


if opts.showLinks
  if nargin >= 2 && ~isempty(inputname(2))  % a variables list was given
    varname = inputname(2) ;
  elseif ~isempty(inputname(1))  % only a Net object was given
    varname = [inputname(1) '.vars'] ;
  else  % unnamed Net object
    varname = 'net.vars' ;
  end
  % note that the links must have a fixed number of characters, otherwise
  % the columns will be misaligned
  link = @(i) sprintf('<a href="matlab:display(''%s{%3d}'')">%s</a>', ...
    varname, i, values{i}) ;
  values = arrayfun(@(i) {link(i)}, 1:numel(values)) ;
end
idx = arrayfun(@(i) {num2str(i)}, 1:2:numel(info)-1) ;

% now print out the info as a table
str = repmat(' ', numel(info) / 2 + 1, 1) ;
str = [str, char('Idx', idx{:})] ;
str(:,end+1:end+2) = ' ' ;
str = [str, char('Function', funcs{1:2:end-1})] ;
str(:,end+1:end+2) = ' ' ;
str = [str, char('Name', info(1:2:end-1).name)] ;

% repeat same set of columns for value and der (size/class/flags/min/max)
headers = {'Value', 'Derivative'} ;

% create dummy links in the headers, to align them nicely
if opts.showLinks
  spaces = numel(['display(''' varname '{   }'')']);
  headers = cellfun(@(name) {[name '<a href="matlab:' blanks(spaces) '"></a>']}, headers);
end

for i = 1:2
  idx = i : 2 : numel(values) - 2 + i ;  % odd or even elements, respectively
  str(:,end+1:end+2) = ' ' ;
  str = [str, char(headers{i}, values{idx})] ;
  str(:,end+1:end+2) = ' ' ;
  str = [str, char('Flags', flags{idx})] ;
  if opts.showRange
    str(:,end+1:end+2) = ' ' ;
    str = [str, char('Min', minStr(idx,:))] ;
    str(:,end+1:end+2) = ' ' ;
    str = [str, char('Max', maxStr(idx,:))] ;
  end
end

disp(str) ;
end
