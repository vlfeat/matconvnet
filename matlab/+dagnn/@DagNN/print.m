function str = display(self, inputs, varargin)
% DISPLAY   Display information about the DAGNN
%    DAGNN.DISPLAY() displays a summary of the functions and parameters in the network.
%
%    DAGNN.DISPLAY(INPUTS) where INPUTS is a cell array of the type
%    {'input1nam', input1size, 'input2name', input2size, ...} prints
%    information using the specified size for each of the listed inputs.
%
%    DAGNN.DISPLAY(..., OPT, VAL, ...) allows specifying the following
%    options:
%
%    All:: false
%       Display all the information below.
%
%    Functions:: true
%       Whether to display the functions.
%
%    Parameters:: true
%       Whether to display the parameters.
%
%    Variables:: false
%       Whether to display the variables.
%
%    Dependencies:: false
%       Whether to display the dependency (geometric transformation)
%       of each variables from each
%       input.
%
%    Format:: 'ascii'
%       Choose between 'ascii', 'latex', and 'csv'.
%
%    See also: DAGNN, DAGNN.GETVARGEOMETRY().

opts.all = false ;
opts.format = 'ascii' ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.functions = true ;
opts.parameters = true ;
opts.variables = opts.all ;
opts.dependencies = opts.all ;
opts.maxNumColumns = 18 ;
opts = vl_argparse(opts, varargin) ;

if nargin == 1, inputs = {} ; end
geom = self.getVarGeometry(inputs) ;
str = {''} ;

if opts.functions
  % print functions
  table = {'func', '-', 'type', 'inputs', 'outputs', 'params', 'pad', 'stride'} ;
  for v = 1:numel(self.params)
    table{v+1,1} = self.layers(v).name ;
    table{v+1,2} = '-' ;
    table{v+1,3} = class(self.layers(v).block) ;
    table{v+1,4} = strtrim(sprintf('%s ', self.layers(v).inputs{:})) ;
    table{v+1,5} = strtrim(sprintf('%s ', self.layers(v).outputs{:})) ;
    table{v+1,6} = strtrim(sprintf('%s ', self.layers(v).params{:})) ;
    table{v+1,7} = pdims(self.layers(v).block.pad) ;
    table{v+1,8} = pdims(self.layers(v).block.stride) ;
  end
  str{end+1} = printtable(opts, table') ;
  str{end+1} = sprintf('\n') ;
end

if opts.parameters
  % print parameters
  table = {'param', '-', 'dims', 'size', 'fanout'} ;
  for v = 1:numel(self.params)
    table{v+1,1} = self.params(v).name ;
    table{v+1,2} = '-' ;
    table{v+1,3} = pdims(size(self.params(v).value)) ;
    table{v+1,4} = pmem(prod(size(self.params(v).value)) * 4) ;
    table{v+1,5} = sprintf('%d',self.params(v).fanout) ;
  end
  str{end+1} = printtable(opts, table') ;
  str{end+1} = sprintf('\n') ;
end

if opts.variables
  % print variables
  table = {'var', '-', 'dims', 'size', 'fanin', 'fanout'} ;
  for v = 1:numel(self.vars)
    table{v+1,1} = self.vars(v).name ;
    table{v+1,2} = '-' ;
    table{v+1,3} = pdims(geom.vars(v).size) ;
    table{v+1,4} = pmem(prod(geom.vars(v).size) * 4) ;
    table{v+1,5} = sprintf('%d',self.vars(v).fanin) ;
    table{v+1,6} = sprintf('%d',self.vars(v).fanout) ;
  end
  str{end+1} = printtable(opts, table') ;
  str{end+1} = sprintf('\n') ;
end

if opts.dependencies
  % print variable to input dependencies
  for i = 1:numel(geom.inputs)
    table = {sprintf('dep on ''%s''', geom.inputs{i}), '-', 'stride', 'offset', 'rec. field'} ;
    for v = 1:numel(geom.vars)
      j = find(strcmp(geom.inputs{i}, {geom.vars(v).transforms.name})) ;
      map = geom.vars(v).transforms(j).map ;
      stride = [map(1,1), map(2,2)] ;
      offset = [map(1,3), map(2,3)] ;
      rf = [map(4,6) - map(1,3) + 1, map(5,6) - map(2,3) + 1] ;
      table{v+1,1} = self.vars(v).name ;
      table{v+1,2} = '-' ;
      table{v+1,3} = pdims(stride) ;
      table{v+1,4} = pdims(offset + rf / 2) ;
      table{v+1,5} = pdims(rf) ;
    end
    str{end+1} = printtable(opts, table') ;
    str{end+1} = sprintf('\n') ;
  end
end

% finish
str = horzcat(str{:}) ;
if nargout == 0,
  fprintf('%s',str) ;
  clear str ;
end

end

% -------------------------------------------------------------------------
function str = printtable(opts, table)
% -------------------------------------------------------------------------
str = {''} ;
for i=2:opts.maxNumColumns:size(table,2)
  sel = i:min(i+opts.maxNumColumns-1,size(table,2)) ;
  str{end+1} = printtablechunk(opts, table(:, [1 sel])) ;
  str{end+1} = sprintf('\n') ;
end
str = horzcat(str{:}) ;
end

% -------------------------------------------------------------------------
function str = printtablechunk(opts, table)
% -------------------------------------------------------------------------
str = {''} ;
switch opts.format
  case 'ascii'
    sizes = max(cellfun(@(x) numel(x), table),[],1) ;
    for i=1:size(table,1)
      for j=1:size(table,2)
        s = table{i,j} ;
        fmt = sprintf('%%%ds|', sizes(j)) ;
        if isequal(s,'-'), s=repmat('-', 1, sizes(j)) ; end
        str{end+1} = sprintf(fmt, s) ;
      end
      str{end+1} = sprintf('\n') ;
    end

  case 'latex'
    sizes = max(cellfun(@(x) numel(x), table),[],1) ;
    str{end+1} = sprintf('\\begin{tabular}{%s}\n', repmat('c', 1, numel(sizes))) ;
    for i=1:size(table,1)
      if isequal(table{i,1},'-'), str{end+1} = sprintf('\\hline\n') ; continue ; end
      for j=1:size(table,2)
        s = table{i,j} ;
        fmt = sprintf('%%%ds', sizes(j)) ;
        str{end+1} = sprintf(fmt, latexesc(s)) ;
        if j<size(table,2), str{end+1} = sprintf('&') ; end
      end
      str{end+1} = sprintf('\\\\\n') ;
    end
    str{end+1}= sprintf('\\end{tabular}\n') ;

  case 'csv'
    sizes = max(cellfun(@(x) numel(x), table),[],1) + 2 ;
    for i=1:size(table,1)
      if isequal(table{i,1},'-'), continue ; end
      for j=1:size(table,2)
        s = table{i,j} ;
        fmt = sprintf('%%%ds,', sizes(j)) ;
        str{end+1} = sprintf(fmt, ['"' s '"']) ;
      end
      str{end+1} = sprintf('\n') ;
    end

  otherwise
    error('Uknown format %s', opts.format) ;
end
str = horzcat(str{:}) ;
end

% -------------------------------------------------------------------------
function s = latexesc(s)
% -------------------------------------------------------------------------
s = strrep(s,'\','\\') ;
s = strrep(s,'_','\char`_') ;
end

% -------------------------------------------------------------------------
function s = pmem(x)
% -------------------------------------------------------------------------
if isnan(x),       s = 'NaN' ;
elseif x < 1024^1, s = sprintf('%.0fB', x) ;
elseif x < 1024^2, s = sprintf('%.0fKB', x / 1024) ;
elseif x < 1024^3, s = sprintf('%.0fMB', x / 1024^2) ;
else               s = sprintf('%.0fGB', x / 1024^3) ;
end
end

% -------------------------------------------------------------------------
function s = pdims(x)
% -------------------------------------------------------------------------
if all(isnan(x))
  s = 'n/a' ;
  return ;
end
if all(x==x(1))
  s = sprintf('%.4g', x(1)) ;
else
  s = sprintf('%.4gx', x(:)) ;
  s(end) = [] ;
end
end
