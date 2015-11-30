function [opts, args] = vl_argparse(opts, args, varargin)
%VL_ARGPARSE Parse list of parameter-value pairs.
%   OPTS = VL_ARGPARSE(OPTS, ARGS) updates the structure OPTS based on
%   the specified parameter-value pairs ARGS={PAR1, VAL1, ... PARN,
%   VALN}. If a parameter PAR cannot be matched to any of the fields
%   in OPTS, the function generates an error.
%
%   One or more of the (PAR, VAL) pairs in the argument list can be
%   replaced by a structure; in this case, the fields of the structure
%   are used as paramater names and the field values as parameter
%   values.
%
%   Parameters that have a struct value in OPTS are processed
%   recursively, updating the individual subfields.  This behaviour
%   can be suppressed by using VL_ARGPARSE(OPTS, ARGS,
%   'nonrecursive'), in which case the struct value is copied directly
%   (hence deleting any existing subfield existing in OPTS). A direct
%   copy occurrs also if the struct value in OPTS is a structure with
%   no fields.
%
%   [OPTS, ARGS] = VL_ARGPARSE(OPTS, ARGS) copies any parameter in
%   ARGS that does not match OPTS back to ARGS instead of producing an
%   error. Options specified as structures are passed back as a list
%   of (PAR, VAL) pairs.
%
%   Example::
%     The function can be used to parse a list of arguments
%     passed to a MATLAB functions:
%
%        function myFunction(x,y,z,varargin)
%        opts.parameterName = defaultValue ;
%        opts = vl_argparse(opts, varargin)
%
%     If only a subset of the options should be parsed, for example
%     because the other options are interpreted by a subroutine, then
%     use the form
%
%        [opts, varargin] = vl_argparse(opts, varargin)
%
%     that copies back to VARARGIN any unknown parameter.
%
%   See also: VL_HELP().

% Copyright (C) 2015 Andrea Vedaldi.
% Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

if ~isstruct(opts), error('OPTS must be a structure') ; end
if ~iscell(args), args = {args} ; end

recursive = true ;
if numel(varargin) == 1
  if strcmp(lower(varargin{1}), 'nonrecursive') ;
    recursive = false ;
  else
    error('Unknown option specified.') ;
  end
end
if numel(varargin) > 1
  error('There can be at most one option.') ;
end

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
  field = params{i} ;
  if ~isfield(opts, field)
    field = findfield(opts, field) ;
  end
  if ~isempty(field)
    % The parameter was found in OPTS
    
    if isstruct(opts.(field))
      % The parameter has a struct value (in OPTS)
      if ~isstruct(values{i})
        error('Cannot assign a non-struct value to the struct parameter ''%s''.', ...
          field) ;
      end
      if recursive && numel(fieldnames(opts.(field))) > 0
        % Process the struct value recursively
        if nargout > 1
          [opts.(field), rest] = vl_argparse(opts.(field), values{i}) ;
          args = horzcat(args, {field, cell2struct(rest(2:2:end), rest(1:2:end), 2)}) ;
        else
          opts.(field) = vl_argparse(opts.(field), values{i}) ;
        end
      else
        % Copy the struct value as is
        opts.(field) = values{i} ;
      end
    else
      % The parameter does not have a struct value (in OPTS)
      % Copy as is
      opts.(field) = values{i} ;
    end
  else
    % The parameter was *not* found in OPTS
    if nargout <= 1
      error('Uknown parameter ''%s''', params{i}) ;
    else
      args = horzcat(args, {params{i}, values{i}}) ;
    end
  end
end

function field = findfield(opts, field)
fields=fieldnames(opts) ;
i=find(strcmpi(fields, field)) ;
if ~isempty(i)
  field=fields{i} ;
else
  field=[] ;
end


