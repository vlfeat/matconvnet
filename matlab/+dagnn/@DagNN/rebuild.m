function rebuild(obj)
%REBUILD Rebuild internal data structures
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
