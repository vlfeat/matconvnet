function y = slice(x, subs)
% Helper function implementing forward indexing operation. The derivative
% is implemented in Layer.eval() for efficiency with sparse updates, so
% this does not correspond to the usual vl_nn* interface.
  y = x(subs{:}) ;
end

