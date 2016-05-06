function display(obj, name)
% DISPLAY(OBJ)
% Overload DISPLAY to show hyperlinks in command window, allowing one to
% interactively traverse the network. Note that the builtin DISP is
% unchanged.

% Copyright (C) 2016 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).


  if nargin < 2
    name = inputname(1) ;
  end
  fprintf('\n%s', name) ;
  
  % non-scalar, use standard display
  if builtin('numel', obj) ~= 1
    fprintf(' =\n\n') ;
    disp(obj) ;
    return
  end

  if numel(name) > 30  % line break for long names
    fprintf('\n')
  end
  if isempty(obj.func)  % other classes, such as Input or Param
    fprintf(' = %s\n\n', class(obj)) ;
  else
    % a standard Layer, expressing a function call
    fprintf(' = %s(', char(obj.func)) ;

    for i = 1:numel(obj.inputs)
      input = obj.inputs{i} ;

      if ~isa(input, 'Layer')
        % use Matlab's native display of single cells, which provides a
        % nice short representation of any object (e.g. '[3x3 double]')
        fprintf(strtrim(evalc('disp({input})'))) ;
      else
        % another layer, display it along with a navigation hyperlink
        if ~isempty(input.name)
          label = input.name ;
        elseif isa(input, 'Input')
          label = 'Input' ;
        elseif isa(input, 'Param')
          label = sprintf('Param(%s)', strtrim(evalc('disp({input.value})'))) ;
        else
          label = sprintf('inputs{%i}', i) ;
        end
        cmd = sprintf('%s.inputs{%i}', name, i) ;

        fprintf('<a href="matlab:display(%s,''%s'')">%s</a>', cmd, cmd, label) ;
      end
      if i < numel(obj.inputs)
        fprintf(', ') ;
      end
    end
    fprintf(')\n\n') ;
  end

  disp(obj) ;

  if ~isempty(obj.source)
    [~, file, ext] = fileparts(obj.source(1).file) ;
    fprintf('Defined in <a href="matlab:opentoline(''%s'',%i)">%s%s, line %i</a>.\n', ...
      obj.source(1).file, obj.source(1).line, file, ext, obj.source(1).line) ;
  end
end