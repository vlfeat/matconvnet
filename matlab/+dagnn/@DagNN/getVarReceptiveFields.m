function rfs = getVarReceptiveFields(obj, var)
%GETVARRECEPTIVEFIELDS Get the receptive field of a variable
%   RFS = GETVARRECEPTIVEFIELDS(OBJ, VAR) gets the receptivie fields
%   RFS of all the variables of the DagNN OBJ into variable VAR.
%   VAR is a variable name or index.
%
%   RFS has the same format as DAGNN.GETRECEPTIVEFIELDS() and one entry for
%   each variable in the DagNN. For example, RFS(i) is the receptive field
%   of the i-th variable in the DagNN into variable VAR. If the i-th
%   variable is not a descendent of VAR in the DAG, then there is no
%   receptive field, indicated by 'rfs(i).size == []'. If the receptive
%   field cannot be computed (e.g. because it depends on the values of
%   variables and not just on the network topology, or if it cannot be
%   expressed as a sliding window), then 'rfs(i).size = [NaN NaN]'.

% Copyright (C) 2015 Karel Lenc and Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

if ~isnumeric(var), var = obj.getVarIndex(var) ; end
nv = numel(obj.vars) ;
nw = numel(var) ;
rfs = struct('size', cell(nw, nv), 'stride', cell(nw, nv), 'offset', cell(nw,nv)) ;

for w = 1:numel(var)
  rfs(w,var(w)).size = [1 1] ;
  rfs(w,var(w)).stride = [1 1] ;
  rfs(w,var(w)).offset = [1 1] ;
end

for l = 1:numel(obj.layers)
  % visit all blocks and get their receptive fields
  in = obj.layers(l).inputIndexes ;
  out = obj.layers(l).outputIndexes ;
  blockRfs = obj.layers(l).block.getReceptiveFields() ;

  for w = 1:numel(var)
    if all(out <= var(w)), continue ; end

    % find receptive fields in each of the inputs of the block
    for i = 1:numel(in)
      for j = 1:numel(out)
        outrf = rfs(out(j)) ;
        rf = composeReceptiveFields(rfs(w, in(i)), blockRfs(i,j)) ;
        rfs(w, out(j)) = resolveReceptiveFields([rfs(w, out(j)), rf]) ;
      end
    end
  end
end
end

% -------------------------------------------------------------------------
function rf = composeReceptiveFields(rf1, rf2)
% -------------------------------------------------------------------------
if isempty(rf1.size)
  rf.size = [] ;
  rf.stride = [] ;
  rf.offset = [] ;
  return ;
end

y1 = rf2.offset(1) - (rf2.size(1)-1)/2 ;
y2 = rf2.offset(1) + (rf2.size(1)-1)/2 ;
x1 = rf2.offset(2) - (rf2.size(2)-1)/2 ;
x2 = rf2.offset(2) + (rf2.size(2)-1)/2 ;

v1 = rf1.offset(1) - (rf1.size(1)-1)/2 + rf1.stride(1) * (y1 - 1) ;
v2 = rf1.offset(1) + (rf1.size(1)-1)/2 + rf1.stride(1) * (y2 - 1) ;
u1 = rf1.offset(2) - (rf1.size(2)-1)/2 + rf1.stride(2) * (x1 - 1) ;
u2 = rf1.offset(2) + (rf1.size(2)-1)/2 + rf1.stride(2) * (x2 - 1) ;

h = v2 - v1 + 1 ;
w = u2 - u1 + 1 ;
rf.size = [h, w] ;
rf.stride = rf1.stride .* rf2.stride ;
rf.offset = [v1+v2,u1+u2]/2 ;
end

% -------------------------------------------------------------------------
function rf = resolveReceptiveFields(rfs)
% -------------------------------------------------------------------------

rf.size = [] ;
rf.stride = [] ;
rf.offset = [] ;

for i = 1:numel(rfs)
  if isempty(rfs(i).size), continue ; end
  if isnan(rfs(i).size)
    rf.size = [NaN NaN] ;
    rf.stride = [NaN NaN] ;
    rf.offset = [NaN NaN] ;
    break ;
  end
  if isempty(rf.size)
    rf = rfs(i) ;
  else
    if ~isequal(rf.offset,rfs(i).stride) || ~isequal(rf.stride,rfs(i).stride)
      % incompatible geometry; this cannot be represented by a sliding
      % receptive field and often denotes an error in the network
      % structure
      rf.size = [NaN NaN] ;
      rf.stride = [NaN NaN] ;
      rf.offset = [NaN NaN] ;
      break;
    else
      rf.size = max(vertcat(rf.size, rfs(i).size),[],1) ;
    end
  end
end
end


