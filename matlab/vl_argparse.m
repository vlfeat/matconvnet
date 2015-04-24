function [opts, args] = vl_argparse(opts, args)
% VL_ARGPARSE  Parse list of parameter-value pairs
%   OPTS = VL_ARGPARSE(OPTS, ARGS) updates the structure OPTS based on
%   the specified parameter-value pairs ARGS={PAR1, VAL1, ... PARN,
%   VALN}. The function produces an error if an unknown parameter name
%   is passed on. Values that are structures are copied recursively.
%
%   Any of the PAR, VAL pairs can be replaced by a structure; in this
%   case, the fields of the structure are used as paramaters and the
%   field values as values.
%
%   [OPTS, ARGS] = VL_ARGPARSE(OPTS, ARGS) copies any parameter in
%   ARGS that does not match OPTS back to ARGS instead of producing an
%   error. Options specified as structures are expaned back to PAR,
%   VAL pairs.
%
%   Example::
%     The function can be used to parse a list of arguments
%     passed to a MATLAB functions:
%
%       function myFunction(x,y,z,varargin)
%       opts.parameterName = defaultValue ;
%       opts = vl_argparse(opts, varargin)
%
%     If only a subset of the options should be parsed, for example
%     because the other options are interpreted by a subroutine, then
%     use the form
%
%      [opts, varargin] = vl_argparse(opts, varargin)
%
%     that copies back to VARARGIN any unknown parameter.
%
%   See also: VL_HELP().

% Authors: Andrea Vedaldi

% Copyright (C) 2015 Andrea Vedaldi.
% Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

if ~isstruct(opts), error('OPTS must be a structure') ; end
if ~iscell(args), args = {args} ; end

% convert ARGS into a structure
ai = 1 ;
params = {} ;
values = {} ;
while ai <= length(args)
  if isstr(args{ai})
    params{end+1} = args{ai} ; ai = ai + 1 ;
    values{end+1} = args{ai} ; ai = ai + 1 ;
  elseif isstruct(args{ai}) ;
    params = horzcat(params, fieldnames(args{ai})') ;
    values = horzcat(values, struct2cell(args{ai})') ;
    ai = ai + 1 ;
  else
    error('Expected either a param-value pair or a structure') ;
  end
end
args = {} ;

% copy parameters in the opts structure, recursively
for i = 1:numel(params)
  if isfield(opts, params{i})
    if isstruct(values{i})
      if ~isstruct(opts.(params{i}))
        error('The value of parameter %d is a structure in the arguments but not a structure in OPT.',params{i}) ;
      end
      if nargout > 1
        [opts.(params{i}), rest] = vl_argparse(opts.(params{i}), values{i}) ;
        args = horzcat(args, {params{i}, cell2struct(rest(2:2:end), rest(1:2:end), 2)}) ;
      else
        opts.(params{i}) = vl_argparse(opts.(params{i}), values{i}) ;
      end
    else
      opts.(params{i}) = values{i} ;
    end
  else
    if nargout <= 1
      error('Uknown parameter ''%s''', params{i}) ;
    else
      args = horzcat(args, {params{i}, values{i}}) ;
    end
  end
end
