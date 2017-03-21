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

  switch mode
  case {'normal', 'forward'}  % forward and backward
    if isfield(net.inputs, 'testMode')
      net.setInputs('testMode', false) ;
    end
  case 'test'  % test mode
    if isfield(net.inputs, 'testMode')
      net.setInputs('testMode', true) ;
    end
  otherwise
    error('Unknown mode ''%s''.', mode) ;
  end

  % use local variables for efficiency
  forward = net.forward ;
  vars = net.vars ;
  net.vars = {} ;  % allows Matlab to release memory when needed

  % forward pass
  for k = 1:numel(forward)
    layer = forward(k) ;
    args = layer.args ;
    args(layer.inputArgPos) = vars(layer.inputVars) ;
    
    out = cell(1, max(layer.outputArgPos)) ;
    [out{:}] = layer.func(args{:}) ;
    
    vars(layer.outputVar) = out(layer.outputArgPos);
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
        inputDers = layer.inputVars + 1 ;  % note this includes incorrect indexes at output der args, but they'll be ignored with FIND
        if layer.accumDer
          for i = find(inputArgPos <= numel(out))
            vars{inputDers(i)} = vars{inputDers(i)} + out{inputArgPos(i)} ;
          end
        else
          % special case, do not accumulate derivatives; used to implement
          % ReLU short-circuiting.
          ii = inputArgPos <= numel(out) ;
          vars(inputDers(ii)) = out(inputArgPos(ii)) ;
        end
      else
        % special case, indexing. the derivative update is sparse.
        % args = {X, I1, I2, ..., DYDZ}, derivative of X(I1, I2, ...).
        inputDer = layer.inputVars(1) + 1 ;  % index of input derivative var
        subs = args(2:end-1) ;  % indexing subscripts
        
        % there's a fast sparse update that doesn't handle repeated
        % indexes, and a slow one that does. to do: MEX file.
        repeats = false ;  % check for repeated indexes
        for i = 1:numel(subs)
          if ~ischar(subs{i}) && any(diff(sort(subs{i})) == 0)  % faster than unique()
            repeats = true ;
            break
          end
        end
        if ~repeats
          % very efficient, but doesn't handle repeated indexes
          if isequal(vars{inputDer}, 0)  % must initialize with the right size and class
            vars{inputDer} = zeros(size(vars{inputDer - 1}), 'like', args{end}) ;
          end
          vars{inputDer}(subs{:}) = vars{inputDer}(subs{:}) + args{end} ;
        else
          % enumerate all indexed elements explicitly to accumulate.
          % replace colon keyword/logical indexing with actual subscripts
          for i = 1:numel(subs)
            if isequal(subs{i}, ':')
              if i < numel(subs)
                subs{i} = 1:size(args{1},i) ;
              else  % special case, last subscripted dimension contains all trailing dimensions
                sz = size(args{1}) ;
                subs{i} = 1:prod(sz(i:end)) ;
              end
            elseif islogical(subs{i})
              subs{i} = find(subs{i}) ;
            end
          end
          subs_ = cell(size(subs));
          [subs_{:}] = ndgrid(subs{:});  % enumerate subscripts of all indexed elements
          ii = sub2ind(size(args{1}), subs_{:});  % convert to linear indexes
          der = accumarray(ii(:), args{end}(:), [numel(args{1}), 1]);  % accumulate gradients
          der = reshape(der, size(args{1}));  % reshape back to tensor
          vars{inputDer} = vars{inputDer} + der ;
        end
      end
    end
  end

  net.vars = vars ;
end
