function vl_simplenn_diagnose(net, res)
% VL_SIMPLENN_DIAGNOSE  Plot diagnostic information
%   VL_SIMPLENN_DIAGNOSE(NET, RES) plots in the current window
%   the average, maximum, and miminum element for all the filters
%   and biases in the network NET. If RES is also provided, it will
%   plot the average, minimum, and maximum element for all the
%   intermediate responses and deriviatives stored in RES as well.
%
%   This function can be used to rapidly glance at the evolution
%   of the paramters during training.

n = numel(net.layers) ;
fmu = NaN + zeros(1, n) ;
fmi = fmu ;
fmx = fmu ;
bmu = fmu ;
bmi = fmu ;
bmx = fmu ;
xmu = fmu ;
xmi = fmi ;
xmx = fmx ;
dxmu = fmu ;
dxmi = fmi ;
dxmx = fmx ;
dfmu = fmu ;
dfmi = fmu ;
dfmx = fmu ;
dbmu = fmu ;
dbmi = fmu ;
dbmx = fmu ;

for i=1:numel(net.layers)
  ly = net.layers{i} ;
  if strcmp(ly.type, 'conv') && numel(ly.filters) > 0
    x = gather(ly.filters) ;
    fmu(i) = mean(x(:)) ;
    fmi(i) = min(x(:)) ;
    fmx(i) = max(x(:)) ;
  end
  if strcmp(ly.type, 'conv') && numel(ly.biases) > 0
    x = gather(ly.biases) ;
    bmu(i) = mean(x(:)) ;
    bmi(i) = min(x(:)) ;
    bmx(i) = max(x(:)) ;
  end
  if nargin > 1
    if numel(res(i).x) > 1
      x = gather(res(i).x) ;
      xmu(i) = mean(x(:)) ;
      xmi(i) = min(x(:)) ;
      xmx(i) = max(x(:)) ;
    end
    if numel(res(i).dzdx) > 1
      x = gather(res(i).dzdx);
      dxmu(i) = mean(x(:)) ;
      dxmi(i) = min(x(:)) ;
      dxmx(i) = max(x(:)) ;
    end
    if ~isempty(res(i).dzdw)
      if strcmp(ly.type, 'conv') && numel(res(i).dzdw{1}) > 0
        x = gather(res(i).dzdw{1}) ;
        dfmu(i) = mean(x(:)) ;
        dfmi(i) = min(x(:)) ;
        dfmx(i) = max(x(:)) ;
      end
      if strcmp(ly.type, 'conv') && numel(res(i).dzdw{2}) > 0
        x = gather(res(i).dzdw{2}) ;
        dbmu(i) = mean(x(:)) ;
        dbmi(i) = min(x(:)) ;
        dbmx(i) = max(x(:)) ;
      end
    end
  end
end

if nargin > 1
  np = 6 ;
else
  np = 2 ;
end

clf ; subplot(np,1,1) ;
errorbar(1:n, fmu, -(fmi-fmu), fmx-fmu, 'bo') ;
grid on ;
xlabel('layer') ;
ylabel('filters') ;
title('coefficient ranges') ;

subplot(np,1,2) ;
errorbar(1:n, bmu, -(bmi-bmu), bmx-bmu, 'bo') ;
grid on ;
xlabel('layer') ;
ylabel('biases') ;

if nargin > 1
  subplot(np,1,3) ;
  errorbar(1:n, xmu, -(xmi-xmu), xmx-xmu, 'bo') ;
  grid on ;
  xlabel('layer') ;
  ylabel('x') ;

  subplot(np,1,4) ;
  errorbar(1:n, dxmu, -(dxmi-dxmu), dxmx-dxmu, 'bo') ;
  grid on ;
  xlabel('layer') ;
  ylabel('dzdx') ;

  subplot(np,1,5) ;
  errorbar(1:n, dfmu, -(dfmi-dfmu), dfmx-dfmu, 'bo') ;
  grid on ;
  xlabel('layer') ;
  ylabel('dfilters') ;

  subplot(np,1,6) ;
  errorbar(1:n, dbmu, -(dbmi-dbmu), dbmx-dbmu, 'bo') ;
  grid on ;
  xlabel('layer') ;
  ylabel('dbiases') ;
end


drawnow ;
