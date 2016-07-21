function dynamic_subplot(ax, aspect, property, region)
%DYNAMIC_SUBPLOT Dynamically reflowing subplots, to maintain aspect ratio.
%   DYNAMIC_SUBPLOT
%   Shows a demo with 30 subplots.
%
%   DYNAMIC_SUBPLOT(AX)
%   Dynamically reflows axes with handles AX when parent figure is
%   resized. The number of rows and columns of the subplots is chosen to
%   maintain a constant subplot aspect ratio as much as possible.
%
%   AX does not have to contain axes handles; other elements such as
%   uicontrol are also supported. In that case change PROPERTY to
%   'Position' (see below).
%
%   DYNAMIC_SUBPLOT(AX, ASPECT)
%   Specifies the desired aspect ratio (width/height). Default is 4/3.
%
%   DYNAMIC_SUBPLOT(AX, ASPECT, PROPERTY)
%   Specifies the name of the property of AX to change. It must accept
%   values as [left, bottom, width, height]. Default is 'OuterPosition'.
%
%   DYNAMIC_SUBPLOT(AX, ASPECT, PROPERTY, REGION)
%   Specifies the total rectangle area of the subplots, in normalized
%   coordinates [left, bottom, width, height]. Default is [0, 0, 1, 1].
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
  if nargin < 4, region = [0, 0, 1, 1] ; end
  
  
  % find first axes' parent figure
  fig = ax(1) ;
  while ~strcmp(get(fig, 'Type'), 'figure') && ~isempty(fig)
    fig = get(fig, 'Parent') ;
  end

  % set its callback
  fcn = @(s, e) resize_fcn(fig, ax, aspect, property, region) ;
  set(fig, 'ResizeFcn', fcn) ;
  fcn() ;
end

function resize_fcn(fig, ax, aspect, property, region)
  n = numel(ax) ;  % number of subplots
  fpos = get(fig, 'Position') ;  % figure size
  fpos(3:4) = fpos(3:4) .* region(3:4) ;  % adjust by relative region size
  
  % test aspect ratios of all possible numbers of columns, vectorized
  cols = 1:n ;
  rows = ceil(n ./ cols) ;  % number of rows to tile N subplots
  asp = fpos(3) .* rows ./ (fpos(4) .* cols) ;  % subplots' aspect ratios
  dist = abs(log(asp) - log(aspect)) ;  % logarithmic distance to preferred aspect ratio
  
  % choose best ratio
  [~, i] = min(dist) ;
  rows = rows(i) ;
  cols = cols(i) ;
  
  % do nothing if the optimal ratio is the current one
  pos = get(ax(1), property) ;
  if pos(3) == region(3) / cols && pos(4) == region(4) / rows
    return
  end
  
  % resize the subplots
  set(ax, 'Units', 'normalized') ;
  k = 1 ;
  for y = 0:rows-1
    for x = 0:cols-1
      if k > n, break ; end
      
      % subplot position, relative to subplots region
      pos = [x / cols, 1 - (y + 1) / rows, 1 / cols, 1 / rows] ;
      
      % adjust it to take into account the subplots region
      pos = [region(1:2) + pos(1:2) .* region(3:4), pos(3:4) .* region(3:4)] ;
      
      set(ax(k), property, pos) ;
      k = k + 1 ;
    end
  end
end

