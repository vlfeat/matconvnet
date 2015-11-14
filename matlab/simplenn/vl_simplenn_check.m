function net = vl_simplenn_check(net)
%VL_SIMPLENN_CHECK  Fix incomplete or outdated NET structure

% move meta information in net.meta subfield
if isfield(net, 'classes')
  net.meta.classes = net.classes ;
  net = rmfield(net, 'classes') ;
end

if isfield(net, 'normalization')
  net.meta.normalization = net.normalization ;
  net = rmfield(net, 'normalization') ;
end

% rename filers, biases into weights{...}
for l = 1:numel(net.layers)
  switch net.layers{l}.type
    case {'conv', 'convt', 'bnorm'}
      if ~isfield(net.layers{l}, 'weights')
        net.layers{l}.weights = {...
          net.layers{l}.filters, ...
          net.layers{l}.biases} ;
        net.layers{l} = rmfield(net.layers{l}, 'filters') ;
        net.layers{l} = rmfield(net.layers{l}, 'biases') ;
      end
  end
end

% add default values for missing fields
for l = 1:numel(net.layers)
  defaults = {} ;
  switch net.layers{l}.type
    case 'conv'
      defaults = {...
        'pad', 0, ...
        'stride', 1} ;
      
    case 'convt'
      defaults = {...
        'crop', 0, ...
        'upsample', 1, ...
        'numGroups', 1} ;
      
    case 'relu'
      defaults = {...
        'leak', 0} ;
  end
  
  for i = 1:2:numel(defaults)
    if ~isfield(net.layers{l}, defaults{i})
      net.layers{l}.(defaults{i}) = defaults{i+1} ;
    end
  end
end


