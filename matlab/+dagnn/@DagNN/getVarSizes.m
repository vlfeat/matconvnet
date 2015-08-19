function sizes = getVarSizes(obj, inputSizes)
%GETVARSIZES  Get the size of the variables
%   SIZES = GETVARSIZES(OBJ, INPUTSIZES) computes the SIZES of the
%   DagNN variables given the size o

nv = numel(obj.vars) ;
sizes = num2cell(NaN(nv, 4),2)' ;

for i = 1:2:numel(inputSizes)
  v = obj.getVarIndex(inputSizes{i}) ;
  sizes{v} = inputSizes{i+1}(:)' ;
end

for layer = obj.layers
  in = layer.inputIndexes ;
  out = layer.outputIndexes ;
  sizes(out) = layer.block.getOutputSizes(sizes(in)) ;
end
