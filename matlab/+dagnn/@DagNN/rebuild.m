function rebuild(obj)
%REBUILD Rebuild the internal data structures of a DagNN object
%   REBUILD(obj) rebuilds the internal data structures
%   of the DagNN obj. It is an helper function used internally
%   to update the network when layers are added or removed.

keep = [obj.vars.fanout] > 0 | [obj.vars.fanin] > 0 ;
obj.vars = obj.vars(keep) ;

keep = [obj.params.fanout] > 0 ;
obj.params = obj.params(keep) ;

obj.varNames = cell2struct(num2cell(1:numel(obj.vars)), {obj.vars.name}, 2) ;
obj.paramNames = cell2struct(num2cell(1:numel(obj.params)), {obj.params.name}, 2) ;
obj.layerNames = cell2struct(num2cell(1:numel(obj.layers)), {obj.layers.name}, 2) ;

for l = 1:numel(obj.layers)
  obj.layers(l).inputIndexes = obj.getVarIndex(obj.layers(l).inputs) ;
  obj.layers(l).outputIndexes = obj.getVarIndex(obj.layers(l).outputs) ;
  obj.layers(l).paramIndexes = obj.getParamIndex(obj.layers(l).params) ;
end

obj.executionOrder = getOrder(obj) ;

% --------------------------------------------------------------------
function order = getOrder(obj)
% --------------------------------------------------------------------
hops = cell(1, numel(obj.vars)) ;
for l = 1:numel(obj.layers)
  for v = obj.layers(l).inputIndexes
    hops{v}(end+1) = l ;
  end
end
order = zeros(1, numel(obj.layers)) ;
for l = 1:numel(obj.layers)
  if order(l) == 0
    order = dagSort(obj, hops, order, l) ;
  end
end
if any(order == -1)
  warning('The network grpah contains a cycle') ;
end
[~,order] = sort(order, 'descend') ;

% --------------------------------------------------------------------
function order = dagSort(obj, hops, order, layer)
% --------------------------------------------------------------------
if order(layer) > 0, return ; end
order(layer) = -1 ; % mark as open
n = 0 ;
for o = obj.layers(layer).outputIndexes ;
  for child = hops{o}
    if order(child) == -1
      return ;
    end
    if order(child) == 0
      order = dagSort(obj, hops, order, child) ;
    end
    n = max(n, order(child)) ;
  end
end
order(layer) = n + 1 ;
