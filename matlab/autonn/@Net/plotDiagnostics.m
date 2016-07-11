function plotDiagnostics(net, numPoints)
%PLOTDIAGNOSTICS
%   NET.PLOTDIAGNOSTICS(NUMPOINTS)
%   Shows a plot of min/max/mean values over tensors, for vars and their
%   derivatives that have diagnostics enabled.
%
%   NUMPOINTS specifies the number of samples in the plot's rolling buffer.
%
%   To enable diagnostics for the output var of a given Layer, set its
%   DIAGNOSTICS property to TRUE.

% Copyright (C) 2016 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  fig = findobj(0, 'Type','figure', 'Tag', 'Net.plotDiagnostics') ;
  if isempty(fig)
    fig = figure() ;
    if isequal(fig, 1) || isequal(get(fig, 'Number'), 1)
      fig = figure() ;  % avoid using figure 1, since it's used by cnn_train
    end
    s = [] ;
  else
    s = get(fig, 'UserData') ;
  end
  n = numel(net.diagnostics) ;

  if ~isstruct(s) || ~isfield(s, 'diagnostics') || ~isequal(s.diagnostics, net.diagnostics) || ...
    ~isequal(s.numPoints, numPoints) || ~all(ishandle([s.ax, s.lines, s.patches]))
    clf ;  % initialize new figure, with n axes
    s = [] ;
    s.diagnostics = net.diagnostics ;
    s.numPoints = numPoints ;
    colors = get(0, 'DefaultAxesColorOrder') ;
    for i = 1:n
      s.ax(i) = axes() ;
      color = colors(mod(floor((i-1)/2), size(colors,1)) + 1, :) ;
      s.lines(i) = line(1:numPoints, NaN(1, numPoints), 'Color', color);

      s.mins{i} = NaN(1, numPoints) ;
      s.maxs{i} = NaN(1, numPoints) ;

      s.patches(i) = patch('XData', [], 'YData', [], 'EdgeColor', 'none', 'FaceColor', color, 'FaceAlpha', 0.5) ;

      set(s.ax(i), 'XLim', [1, numPoints], 'XTickLabel', {''}, 'FontSize', 8) ;
      title(strrep(net.diagnostics(i).name, '_', '\_'), 'FontSize', 9, 'FontWeight', 'normal') ;
    end
    dynamic_subplot(s.ax, 3) ;
    set(fig, 'Name', 'Diagnostics', 'NumberTitle','off', ...
      'Tag', 'Net.plotDiagnostics', 'UserData', s) ;
  end

  % add new points and roll buffer
  for i = 1:n
    data = real(net.vars{net.diagnostics(i).var}(:)) ;
    ps = get(s.lines(i), 'YData') ;
    ps = [ps(2:end), gather(mean(data))] ;
    set(s.lines(i), 'YData', ps) ;

    if ~all(isnan(ps))
      if ~isscalar(data)
        mi = gather(min(data)) ;
        ma = gather(max(data)) ;
        if mi ~= ma
          s.mins{i} = [s.mins{i}(2:end), mi] ;
          s.maxs{i} = [s.maxs{i}(2:end), ma] ;

          valid = find(~isnan(s.mins{i})) ;
          set(s.patches(i), 'XData', [valid, valid(end:-1:1)], ...
            'YData', [s.mins{i}(valid), s.maxs{i}(valid(end:-1:1))]) ;
          set(s.ax(i), 'YLim', [min(s.mins{i}), max(s.maxs{i})]) ;
        end
      else
        set(s.ax(i), 'YLim', [min(ps), max(ps) + 0.01 * max(0.01, abs(max(ps)))]) ;
      end
    end
  end
  set(fig, 'UserData', s) ;
  drawnow ;
end