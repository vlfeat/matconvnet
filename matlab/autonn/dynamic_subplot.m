function dynamic_subplot(ax, aspect, property)
%DYNAMIC_SUBPLOT Dynamically reflowing subplots, to maintain aspect ratio.
%   DYNAMIC_SUBPLOT(AX)
%   Dynamically reflows axes with handles AX when parent figure FIG is
%   resized. The number of rows and columns of the subplots is chosen to
%   maintain a constant subplot aspect ratio as much as possible.
%
%   DYNAMIC_SUBPLOT(AX, ASPECT)
%   Specifies the desired aspect ratio (width/height). Default is 4/3.
%
%   DYNAMIC_SUBPLOT(AX, ASPECT, PROPERTY)
%   Specifies the name of the property of AX to change. It must accept
%   values as [left, top, width, height]. Default is 'OuterPosition'.
%
%   Joao F. Henriques, 2016

  % demo
  if nargin == 0
    figure() ;
    for p = 1:30
        ax(p) = axes('FontSize', 8) ;
        line(1:20, cumsum(randn(20, 1)), 'Color', hsv2rgb(rand(), 0.6, 0.8)) ;
        grid on ;
    end
  end

  if nargin < 2, aspect = 4/3 ; end
  if nargin < 3, property = 'OuterPosition' ; end
  
  
  % find first axes' parent figure
  fig = ax(1) ;
  while ~strcmp(get(fig, 'Type'), 'figure') && ~isempty(fig)
    fig = get(fig, 'Parent') ;
  end

  % set its callback
  fcn = @(s, e) resize_fcn(fig, ax, aspect, property) ;
  set(fig, 'ResizeFcn', fcn) ;
  fcn() ;
end

function resize_fcn(fig, ax, aspect, property)
  n = numel(ax) ;  % number of subplots
  pos = get(fig, 'Position') ;  % figure size
  
  % test aspect ratios of all possible numbers of columns, vectorized
  cols = 1:n ;
  rows = ceil(n ./ cols) ;  % number of rows to tile N subplots
  asp = pos(3) .* rows ./ (pos(4) .* cols) ;  % subplots' aspect ratios
  dist = abs(log(asp) - log(aspect)) ;  % logarithmic distance to preferred aspect ratio
  
  % choose best ratio
  [~, i] = min(dist) ;
  rows = rows(i) ;
  cols = cols(i) ;
  
  % do nothing if the optimal ratio is the current one
  pos = get(ax(1), property) ;
  if pos(3) == 1 / cols && pos(4) == 1 / rows
    return
  end
  
  % resize the subplots
  set(ax, 'Units', 'normalized') ;
  k = 1 ;
  for y = 0:rows-1
    for x = 0:cols-1
      if k > n, break ; end
      set(ax(k), property, [x / cols, 1 - (y + 1) / rows, 1 / cols, 1 / rows]) ;
      k = k + 1 ;
    end
  end
end

