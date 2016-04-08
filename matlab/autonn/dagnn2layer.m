function layer = dagnn2layer(dag)
%DAGNN2LAYER Summary of this function goes here
%   Detailed explanation goes here

  for i = dag.executionOrder
    block = dag.layers(i).block ;
    
    if isa(block, 'dagnn.Conv')
      
      
    elseif isa(block, 'dagnn.Pool')
      
    else
      error(['Unknown block type ''' class(block) ''.']) ;
    end
  end

end

