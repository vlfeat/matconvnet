classdef RNN2D < dagnn.Layer
    % 2D RNN Layer (Dag-RNN)
    % Equations:
    %   a(t) = W*h(t-1) + U*x(t) + b;
    %   h(t) = tanh(a(t));
    
    properties
        % Remark:
        %   if derFnInput == 'x', [derInputs, derParams] = derFn(x, dy)
        %   if derFnInput == 'y', [derInputs, derParams] = derFn(y, dy)
        %   Saves computation when the derivative can be expressed as
        %   a function of its output y
        %   if conserveMemory == true, intermediate results h and a will
        %   not be stored in outputs during forward pass
        % hiddenFnParam for hiddenFn:
        %   ReLU: leak
        
        hiddenDim = 0; % dimension of hidden units
        inputDim = 0;
        dataType = 'single';
        forwardDir = [1 1]; % forward direction
        derFnInput = 'y';
        hiddenFn = 'tanh';
        hiddenFnParam = 0; % equals leak, when hiddenFn == 'relu'
        scalePredecessor = true;
        clip = {'method', 'clip', 'threshold', 10};
        opts = {};
    end
    
    properties (Transient)
        aux = {}
    end
    
    methods
        function outputs = forward(obj, inputs, params)
            % Input and output arguments:
            %    inputs{1}: x
            %    inputs{2}: h0
            %    params{1}: U
            %    params{2}: W
            %    params{3}: b
            %   outputs{1}: h
            
            useGpu = false;
            if isa(inputs{1}, 'gpuArray')
                useGpu = true;
                inputs = cellfun(@(x) gather(x), inputs, 'UniformOutput', false);
                params = cellfun(@(x) gather(x), params, 'UniformOutput', false);
            end
            [outputs{1}] = vl_nnrnnforward(inputs{1}, inputs{2}, ...
                params{1}, params{2}, params{3}, ...
                'forwardDir', obj.forwardDir, ...
                'hiddenFn', obj.hiddenFn, ...
                'derFnInput', obj.derFnInput, ...
                'scalePredecessor', obj.scalePredecessor);
            obj.aux{1} = outputs{1};
            if useGpu
                outputs = cellfun(@(x) gpuArray(x), outputs, 'UniformOutput', false);
            end
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            % Input and output arguments:
            %    inputs{1}: x
            %    inputs{2}: h0
            %    params{1}: U
            %    params{2}: W
            %    params{3}: b
            
            useGpu = false;
            if isa(inputs{1}, 'gpuArray')
                useGpu = true;
                inputs = cellfun(@(x) gather(x), inputs, 'UniformOutput', false);
                params = cellfun(@(x) gather(x), params, 'UniformOutput', false);
                derOutputs = cellfun(@(x) gather(x), derOutputs, 'UniformOutput', false);
            end
            [derInputs{1}, derInputs{2}, derParams{1}, derParams{2}, derParams{3}] = ...
                vl_nnrnnbackward(inputs{1}, inputs{2}, params{1}, params{2}, ...
                [], obj.aux{1}, derOutputs{1}, ...
                'forwardDir', obj.forwardDir, ...
                'hiddenFn', obj.hiddenFn, ...
                'derFnInput', obj.derFnInput, ...
                'scalePredecessor', obj.scalePredecessor);
            % Gradient clipping
            derParams = vl_gradclip(derParams, obj.clip{:});
            if useGpu
                derInputs = cellfun(@(x) gpuArray(x), derInputs, 'UniformOutput', false);
                derParams = cellfun(@(x) gpuArray(x), derParams, 'UniformOutput', false);
            end
        end
        
        function outputSizes = getOutputSizes(obj, inputSizes, paramSizes)
            outputSizes{1} = [inputSizes{1}(1:2) obj.hiddenDim inputSizes{1}(4)];
        end
        
        function params = initParams(obj)
            % Parameters:
            %    params{1}: U
            %    params{2}: W
            %    params{3}: b
            
            dimx = obj.inputDim;
            dimh = obj.hiddenDim;
            dtype = obj.dataType;
            
            % Xavier improved
            scU = sqrt(2 / (dimh*dimx));
            % scW = sqrt(2 / (dimh*dimh));
            
            params{1} = randn([dimh dimx], dtype) * scU;
            % params{2} = randn([dimh dimh], dtype) * scW;
            params{2} = eye(dimh, dtype);
            params{3} = zeros([dimh 1], dtype);
        end
        
        function obj = RNN2D(varargin)
            obj.load(varargin);
            if strcmpi(obj.derFnInput, 'x'), error('`derFnInput` ''x'' not supported yet'); end
            
            % Nonlinear transfer function for hidden units
            if strcmpi(obj.derFnInput, 'x')
                switch lower(obj.hiddenFn)
                    case 'tanh'
                        obj.hiddenFn = @nnfun.tanh;
                    case 'relu'
                        obj.hiddenFn = @(varargin) ...
                            nnfun.relu(varargin{:}, 'leak', obj.hiddenFnParam);
                    case 'sigmoid'
                        obj.hiddenFn = @nnfun.sigmoid;
                end
            else % obj.derFnInput == 'y'
                switch lower(obj.hiddenFn)
                    case 'tanh'
                        obj.hiddenFn = @nnfun.tanhy;
                    case 'relu'
                        obj.hiddenFn = @(varargin) ...
                            nnfun.reluy(varargin{:}, 'leak', obj.hiddenFnParam);
                    case 'sigmoid'
                        obj.hiddenFn = @nnfun.sigmoidy;
                end
            end
        end
    end
end
