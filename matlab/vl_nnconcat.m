function y = vl_nnconcat(inputs, dim, dzdy, varargin)
% VL_NNCONCAT   CNN concatenate multiple inputs
%    Y = VL_NNCONCAT(INPUTS, DIM) concatenates the inputs in the cell
%    array INPUTS along dimension DIM generating an output Y.
%
%    DZDINPUTS = VL_NNCONCAT(INPUTS, DIM, DZDY) computes the
%    derivatives of the function projected to DZDY.

opts.inputSizes = [] ;
opts = vl_argparse(opts, varargin) ;

if nargin < 2, dim = 3; end;
if nargin < 3, dzdy = []; end;

if isempty(dzdy)
  y = cat(dim, inputs{:});
else
  if isempty(opts.inputSizes)
    opts.inputSizes = cellfun(@size, inputs, 'UniformOutput', false) ;
  end
  start = 1 ;
  y = cell(1, numel(opts.inputSizes)) ;
  s.type = '()' ;
  s.subs = {':', ':', ':', ':'} ;
  for i = 1:numel(opts.inputSizes)
    stop = start + opts.inputSizes{i}(dim) ;
    s.subs{dim} = start:stop-1 ; ;
    y{i} = subsref(dzdy,s) ;
    start = stop ;
  end
end
