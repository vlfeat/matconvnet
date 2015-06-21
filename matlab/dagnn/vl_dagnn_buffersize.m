function [ cpum, gpum ] = vl_dagnn_buffersize( res, dzdw )
cpum = 0; gpum = 0;
for ri = 1:numel(res)
  [cpum, gpum] = getsize(res(ri).x, cpum, gpum);
  [cpum, gpum] = getsize(res(ri).dzdx, cpum, gpum);
end
if exist('dzdw', 'var')
  for di = 1:numel(dzdw)
    [cpum, gpum] = getsize(dzdw{di}, cpum, gpum);
  end
end

function [cpum, gpum] = getsize(val, cpum, gpum)
mult = 4;
if isa(val, 'double'), mult = 8 ; end
if ~iscell(val), val = {val}; end;
for vi = 1:numel(val)
  if isa(val{vi},'gpuArray')
    gpum = gpum + mult * numel(val{vi}) ;
  else
    cpum = cpum + mult * numel(val{vi}) ;
  end
end
