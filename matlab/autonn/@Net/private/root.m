function out = root(varargin)
%ROOT
%   Root layer for networks with multiple inputs (see BUILD).
%
%   The derivative is implemented in AUTONN_DER; it simply distributes the
%   output derivative to the inputs. As such, for consistency this layer
%   should reshape and concatenate the inputs into a single vector or
%   tensor; however this is not necessary, as its output is not used.

  out = [] ;

end

