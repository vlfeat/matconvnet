classdef GenericLayer < handle
  properties
    inputs;
    name;
  end
  
  properties (Constant)
    type = 'custom';
  end
  
  % Prototypes
  methods (Static)
    top_data = forward(obj, bottom_data, top_data)
    bottom_data = backward(obj, bottom_data, top_data )
    test(gpu);
  end
  
  methods
    function move(obj, destination)
      % Empty implementation of move
    end;
    
    function [obj, args] = argparse(obj, args, varargin)
      % Same as VL_ARGPARSE but accepts objects

      if numel(varargin) > 0, args = [{args}, varargin] ; end
      
      remainingArgs = {} ;
      names = fieldnames(obj) ;
      
      if mod(length(args),2) == 1
        error('Parameter-value pair expected (missing value?).') ;
      end
      
      for ai = 1:2:length(args)
        paramName = args{ai} ;
        if ~ischar(paramName)
          error('The name of the parameter number %d is not a string.', ...
            (ai-1)/2+1) ;
        end
        value = args{ai+1} ;
        if isfield(obj,paramName)
          obj.(paramName) = value ;
        else
          % try case-insensitive
          i = find(strcmpi(paramName, names)) ;
          if isempty(i)
            if nargout < 2
              error('Unknown parameter ''%s''.', paramName) ;
            else
              remainingArgs(end+1:end+2) = args(ai:ai+1) ;
            end
          else
            obj.(names{i}) = value ;
          end
        end
      end      
      args = remainingArgs ;
    end
  end
end

