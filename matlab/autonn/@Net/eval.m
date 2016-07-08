function eval(net, mode, derOutput, accumulateParamDers)
%EVAL Compute network outputs and/or derivatives.

% Copyright (C) 2016 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).


  if nargin < 2
    mode = 'normal' ;
  end
  if nargin < 3
    derOutput = single(1) ;
  end
  if nargin < 4
    accumulateParamDers = false ;
  end

  % use local variables for efficiency
  vars = net.vars ;
  net.vars = {} ;  % allows Matlab to release memory when needed

  switch mode
  case {'normal', 'forward'}  % forward and backward
    forward = net.forward ;   %#ok<*PROPLC> % disable MLint's complaints
  case 'test'  % test mode
    forward = net.test ;
  otherwise
    error('Unknown mode ''%s''.', mode) ;
  end

  % forward pass
  for k = 1:numel(forward)
    layer = forward(k) ;
    args = layer.args ;
    args(layer.inputArgPos) = vars(layer.inputVars) ;
    vars{layer.outputVar} = layer.func(args{:}) ;
  end

  % backward pass
  if strcmp(mode, 'normal')
    % clear all derivatives. derivatives are even-numbered vars.
    clear = repmat([false; true], numel(vars) / 2, 1);
    if accumulateParamDers  % except for params (e.g. to implement sub-batches)
      clear([net.params.var] + 1) = false ;  % next var is the derivative
    end
    [vars(clear)] = deal({0}) ;

    % set root layer's output derivative
    assert(~isempty(derOutput), 'Must specify non-empty output derivatives for normal mode.')
    vars{end} = derOutput ;

    backward = net.backward ;

    for k = 1:numel(backward)
      % populate function arguments with input vars and derivatives
      layer = backward(k) ;
      args = layer.args ;
      inputArgPos = layer.inputArgPos ;
      args(inputArgPos) = vars(layer.inputVars) ;

      if ~isequal(layer.func, @autonn_slice)
        % call function and collect outputs
        out = cell(1, layer.numInputDer) ;
        [out{:}] = layer.func(args{:}) ;

        % sum derivatives. the derivative var corresponding to each
        % input comes right next to it in the vars list. note that some
        % outputs may be ignored (because they're not input layers,
        % just constant arguments).
        inputDers = layer.inputVars(1:end-1) + 1 ;  % last input is dzdy, doesn't count
        if layer.accumDer
          for i = find(inputArgPos <= numel(out))
            vars{inputDers(i)} = vars{inputDers(i)} + out{inputArgPos(i)} ;
          end
        else
          % special case, do not accumulat derivatives; used to implement
          % ReLU short-circuiting.
          ii = inputArgPos <= numel(out) ;
          vars(inputDers(ii)) = out(inputArgPos(ii)) ;
        end
      else
        % special case, indexing. the derivative update is sparse.
        % args = {input, slicing indexes, output derivative}.
        inputDer = layer.inputVars(1) + 1 ;  % index of input derivative var
        if isequal(vars{inputDer}, 0)  % must initialize with the right size
          vars{inputDer} = zeros(size(vars{inputDer - 1}), 'like', vars{inputDer - 1}) ;
        end
        vars{inputDer}(args{2}{:}) = vars{inputDer}(args{2}{:}) + args{3} ;
      end
    end
  end

  net.vars = vars ;
end
