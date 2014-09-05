function [conf, args] = vl_argparse(conf, args)
% VL_ARGPARSE  Parse list of parameter-value pairs
%   CONF = VL_ARGPARSE(CONF, ARGS) updates the structure CONF based on
%   the specified parameter-value pairs ARGS={PAR1, VAL1, ... PARN,
%   VALN}. The function produces an error if an unknown parameter name
%   is passed in.
%
%   [CONF, ARGS] = VL_ARGPARSE(CONF, ARGS) copies any parameter in
%   ARGS that does not match CONF back to ARGS instead of producing an
%   error.
%
%   Example::
%     The function can be used to parse a list of arguments
%     passed to a MATLAB functions:
%
%       function myFunction(x,y,z,varargin)
%       conf.parameterName = defaultValue ;
%       conf = vl_argparse(conf, varargin)
%
%     If only a subset of the options should be parsed, for example
%     because the other options are interpreted by a subroutine, then
%     use the form
%
%      [conf, varargin] = vl_argparse(conf, varargin)
%
%     that copies back to VARARGIN any unknown parameter.
%
%   See also: VL_OVERRIDE(), VL_HELP().

% Authors: Andrea Vedaldi

% Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

if ~isstruct(conf), error('CONF must be a structure') ; end

remainingArgs = {} ;
names = fieldnames(conf) ;

ai = 1 ;
while ai <= length(args)
  paramName = args{ai} ;
  if isstruct(paramName)
    moreArgs = cat(2, fieldnames(args{ai}), struct2cell(args{ai}))' ;
    [conf,r] = vl_argparse(conf, moreArgs(:)) ;
    remainingArgs = cat(2, remainingArgs, r) ;
    ai = ai +1 ;
    continue ;
  end
  if ~ischar(paramName)
    error('The name of the parameter number %d is not a string nor a structure', (ai-1)/2+1) ;
  end
  if ai + 1 > length(args)
    error('Parameter-value pair expected (missing value?).') ;
  end
  value = args{ai+1} ;
  i = find(strcmpi(paramName, names)) ;
  if isempty(i)
    if nargout < 2
      error('Unknown parameter ''%s''.', paramName) ;
    else
      remainingArgs(end+1:end+2) = args(ai:ai+1) ;
    end
  else
    paramName = names{i} ;
    if isstruct(conf.(paramName))
      [conf.(paramName),r] = vl_argparse(conf.(paramName), {value}) ;
    else
      conf.(paramName) = value ;
    end
  end
  ai = ai + 2 ;
end

args = remainingArgs ;
