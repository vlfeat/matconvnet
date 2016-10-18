function print(net)
%PRINT
%   Displays the network topology in a PDF. Requires DOT.

% Copyright (C) 2016 Karel Lenc, Andrea Vedaldi, Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  % gather all layers
  layers = net.find() ;
  
  % ensure non-empty, unique names
  names = cellfun(@(o) {o.name}, layers) ;
  assert(~any(cellfun('isempty', names)), ...
    'PRINT requires all layers to have names (e.g., with LAYER.SEQUENTIALNAMES).') ;
  assert(numel(unique(names)) == numel(names), ...
    'PRINT requires all layers to have unique names.') ;

  str = {} ;
  str{end+1} = sprintf('digraph DagNN {\n\tfontsize=12\n') ;
  font_style = 'fontsize=12 fontname="helvetica"';
  
  for k = 1:numel(layers)
    layer = layers{k} ;
    if isa(layer, 'Param')
      % Param
      sz = size(layer.value) ;
      label=sprintf('{{%s} | {%s | %s }}', layer.name, pdims(sz), pmem(4*prod(sz))) ;
      str{end+1} = sprintf('\t%s [label="%s" shape=record style="solid,rounded,filled" color=lightsteelblue4 fillcolor=lightsteelblue %s ]\n', ...
        layer.name, label, font_style) ;
    else
      % other layers
      if ~isempty(layer.func)
        func = func2str(layer.func) ;
      else
        func = class(layer) ;
      end
      label = sprintf('{ %s | %s }', layer.name, func) ;
      str{end+1} = sprintf('\t%s [label="%s" shape=record style="bold,filled" color="tomato4" fillcolor="tomato" %s ]\n', ...
        layer.name, label, font_style) ;
    
      for i = 1:numel(layer.inputs)
        if isa(layer.inputs{i}, 'Layer')
          weight = 10 ;
          if isa(layer.inputs{i}, 'Param')
            weight = 1 ;
          end
          str{end+1} = sprintf('\t%s->%s [weight=%i]\n', ...
            layer.inputs{i}.name, layer.name, weight) ;
        end
      end
    end
  end

  str{end+1} = sprintf('}\n') ;
  str = cat(2,str{:}) ;
  
  displayDot(str) ;
end

function displayDot(str)
  %mwdot = fullfile(matlabroot, 'bin', computer('arch'), 'mwdot') ;
  dotexe = 'dot' ;

  in=[tempname '.dot'];
  out=[tempname '.pdf'];

  f = fopen(in,'w') ; fwrite(f, str) ; fclose(f) ;

  cmd = sprintf('"%s" -Tpdf -o "%s" "%s"', dotexe, out, in) ;
  [status, result] = system(cmd) ;
  if status ~= 0
    error('Unable to run %s\n%s', cmd, result) ;
  end
  if ~isempty(result)
    fprintf('Dot output:\n%s\n', result) ;
  end

  %f = fopen(out,'r') ; file=fread(f, 'char=>char')' ; fclose(f) ;
  switch computer
    case 'MACI64'
      system(sprintf('open "%s"', out)) ;
    case 'GLNXA64'
      % start with most generic command, to older/more specific.
      % notice the library path is cleared beforehand, to prevent clashes
      % with some applications.
      commands = {'xdg-open', 'gvfs-open', 'gnome-open', 'kde-open', 'display'};
      for i = 1:numel(commands)
        result = system(sprintf('LD_LIBRARY_PATH= xdg-open "%s"', out)) ;
        if result == 0, break; end
      end
%       system(sprintf('display "%s"', out)) ;
    case 'PCWIN64'
      winopen(out) ;
    otherwise
      fprintf('PDF figure saved at "%s"\n', out) ;
  end
end

function s = pmem(x)
  if isnan(x),       s = 'NaN' ;
  elseif x < 1024^1, s = sprintf('%.0fB', x) ;
  elseif x < 1024^2, s = sprintf('%.0fKB', x / 1024) ;
  elseif x < 1024^3, s = sprintf('%.0fMB', x / 1024^2) ;
  else               s = sprintf('%.0fGB', x / 1024^3) ;
  end
end

function s = pdims(x)
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
