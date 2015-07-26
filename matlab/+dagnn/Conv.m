classdef Conv < dagnn.Filter
  properties
    size = [0 0 0 0]
    hasBias = true
    opts = {'cuDNN'}
  end

  methods
    % ---------------------------------------------------------------------
    function outputs = forward(obj, inputs, params)
      if obj.hasBias
        outputs{1} = vl_nnconv(...
          inputs{1}, params{1}, params{2}, ...
          'pad', obj.pad, ...
          'stride', obj.stride, ...
          obj.opts{:}) ;
      else
        outputs{1} = vl_nnconv(...
          inputs{1}, params{1}, [], ...
          'pad', obj.pad, ...
          'stride', obj.stride, ...
          obj.opts{:}) ;
      end
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      if obj.hasBias
        [derInputs{1}, derParams{1}, derParams{2}] = vl_nnconv(...
          inputs{1}, params{1}, params{2}, derOutputs{1}, ...
          'pad', obj.pad, ...
          'stride', obj.stride, ...
          obj.opts{:}) ;
      else
        [derInputs{1}, derParams{1}] = vl_nnconv(...
          inputs{1}, params{1}, [], derOutputs{1}, ...
          'pad', obj.pad, ...
          'stride', obj.stride, ...
          obj.opts{:}) ;
      end
    end

    % ---------------------------------------------------------------------
    function params = init(obj)
      sc = sqrt(2 / prod(obj.size(1:3))) ;
      params{1} = randn(obj.size,'single') * sc ;
      params{2} = zeros(obj.size(4),1,'single') * sc ;
    end

    %     function [outputSizes, transforms] = forwardGeometry(obj, inputSizes, paramSizes)
    %       transforms{1} = [...
    %         obj.stride(1), 0, 1 - obj.pad(1) - obj.stride(1) 0 0 0 ;
    %         0, obj.stride(2), 1 - obj.pad(3) - obj.stride(2) 0 0 0 ;
    %         0, 0, 1, 0, 0, 0, ;
    %         0, 0, 0, obj.stride(1), 0, 1 - obj.pad(1) - obj.stride(1) + paramSizes{1}(1) - 1 ;
    %         0, 0, 0, 0, obj.stride(2), 1 - obj.pad(3) - obj.stride(2) + paramSizes{1}(2) - 1
    %         0, 0, 0, 0, 0, 1] ;
    %       outputSizes{1}(1:2) = floor((inputSizes{1}(1:2) + obj.pad([1 3]) + obj.pad([2 4]) - paramSizes{1}(1:2)) ./ obj.stride(:)') + 1 ;
    %       outputSizes{1}(3) = paramSizes{1}(4) ;
    %       outputSizes{1}(4) = inputSizes{1}(4) ;
    %     end

    % ---------------------------------------------------------------------
    function obj = Conv(varargin)
      obj.load(varargin) ;
    end

  end
end
