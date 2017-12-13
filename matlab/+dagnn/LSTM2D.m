classdef LSTM2D < dagnn.Layer
    % 2D LSTM Layer (Dag-LSTM)
    %
    % Equations:
    %  Normal LSTM:
    %   a(t) = tanh(Wc * x(t) + Uc * h(t-1) + bc)
    %   i(t) = sigm(Wi * x(t) + Ui * h(t-1) + bi)
    %   f(t) = sigm(Wf * x(t) + Uf * h(t-1) + bf)
    %   o(t) = sigm(Wo * x(t) + Uo * h(t-1) + bo)
    %   c(t) = i(t) .* a(t) + f(t) .* c(t-1)
    %   h(t) = o(t) .* tanh(c(t))
    %  Coupled i and o:
    %   a(t) = tanh(Wc * x(t) + Uc * h(t-1) + bc)
    %   f(t) = sigm(Wf * x(t) + Uf * h(t-1) + bf)
    %   o(t) = sigm(Wo * x(t) + Uo * h(t-1) + bo)
    %   c(t) = f(t) .* c(t-1) + (1-f(t)) .* a(t)
    %   h(t) = o(t) .* tanh(c(t))
    %  LSTM with peep holes:
    %   a(t) = tanh(Wc * x(t) + Uc * h(t-1) + bc)
    %   i(t) = sigm(Wi * x(t) + Ui * h(t-1) + Vi * c(t-1) + bi)
    %   f(t) = sigm(Wf * x(t) + Uf * h(t-1) + Vf * c(t-1) + bf)
    %   o(t) = sigm(Wo * x(t) + Uo * h(t-1) + Vo * c(t) + bo)
    %   c(t) = i(t) .* a(t) + f(t) .* c(t-1)
    %   h(t) = o(t) .* tanh(c(t))
    %  GRU:
    %   z(t) = sigm(Wz * x(t) + Uz * h(t-1) + bz)
    %   r(t) = sigm(Wr * x(t) + Ur * h(t-1) + br)
    %   a(t) = tanh(Wa * x(t) + Ua * (r(t) .* h(t-1)))
    %   h(t) = (1-z(t)) .* h(t-1) + z(t) .* a(t)
    
    % Inputs, Outputs, Params:
    %  Normal LSTM:
    %    inputs{ 1}: x
    %    inputs{ 2}: h0
    %    inputs{ 3}: c0
    %    params{ 1}: Wc
    %    params{ 2}: Wi
    %    params{ 3}: Wf
    %    params{ 4}: Wo
    %    params{ 5}: Uc
    %    params{ 6}: Ui
    %    params{ 7}: Uf
    %    params{ 8}: Uo
    %    params{ 9}: bc
    %    params{10}: bi
    %    params{11}: bf
    %    params{12}: bo
    %   outputs{ 1}: h
    %       aux{ 1}: h
    %       aux{ 2}: c
    %       aux{ 3}: a
    %       aux{ 4}: i
    %       aux{ 5}: f
    %       aux{ 6}: o
    %  Coupled i and o:
    %    inputs{ 1}: x
    %    inputs{ 2}: h0
    %    inputs{ 3}: c0
    %    params{ 1}: Wc
    %    params{ 2}: Wf
    %    params{ 3}: Wo
    %    params{ 4}: Uc
    %    params{ 5}: Uf
    %    params{ 6}: Uo
    %    params{ 7}: bc
    %    params{ 8}: bf
    %    params{ 9}: bo
    %   outputs{ 1}: h
    %       aux{ 1}: h
    %       aux{ 2}: c
    %       aux{ 3}: a
    %       aux{ 4}: f
    %       aux{ 5}: o
    %  LSTM with peep hole:
    %    inputs{ 1}: x
    %    inputs{ 2}: h0
    %    inputs{ 3}: c0
    %    params{ 1}: Wc
    %    params{ 2}: Wi
    %    params{ 3}: Wf
    %    params{ 4}: Wo
    %    params{ 5}: Uc
    %    params{ 6}: Ui
    %    params{ 7}: Uf
    %    params{ 8}: Uo
    %    params{ 9}: Vi
    %    params{10}: Vf
    %    params{11}: Vo
    %    params{12}: bc
    %    params{13}: bi
    %    params{14}: bf
    %    params{15}: bo
    %   outputs{ 1}: h
    %       aux{ 1}: h
    %       aux{ 2}: c
    %       aux{ 3}: a
    %       aux{ 4}: i
    %       aux{ 5}: f
    %       aux{ 6}: o
    %  GRU:
    %    inputs{ 1}: x
    %    inputs{ 2}: h0
    %    params{ 1}: Wz
    %    params{ 2}: Wr
    %    params{ 3}: Wa
    %    params{ 4}: Uz
    %    params{ 5}: Ur
    %    params{ 6}: Ua
    %    params{ 7}: bz
    %    params{ 8}: br
    %    params{ 9}: ba
    %   outputs{ 1}: h
    %       aux{ 1}: h
    %       aux{ 2}: z
    %       aux{ 3}: r
    %       aux{ 4}: a
    
    properties
        style = 'normal';
        hiddenDim = 0; % dimension of hidden units
        inputDim = 0;
        dataType = 'single';
        forwardDir = [1 1]; % forward direction
        scalePredecessor = true;
        clip = {'method', 'clip', 'threshold', 10};
        opts = {};
    end
    
    properties (Transient)
        aux = {}
    end
    
    methods
        function outputs = forward(obj, inputs, params)
            useGpu = false;
            if isa(inputs{1}, 'gpuArray')
                useGpu = true;
                inputs = cellfun(@(x) gather(x), inputs, 'UniformOutput', false);
                params = cellfun(@(x) gather(x), params, 'UniformOutput', false);
            end
            if strcmpi(obj.style, 'normal')
                [outputs{1}, obj.aux{2}, obj.aux{3}, obj.aux{4}, obj.aux{5}, obj.aux{6}] = ...
                    vl_nnlstmforward(inputs{1}, inputs{2}, inputs{3}, ... x, h0, c0
                    params{1}, params{2}, params{3}, params{4}, ... Wc, Wi, Wf, Wo
                    params{5}, params{6}, params{7}, params{8}, ... Uc, Ui, Uf, Uo
                    params{9}, params{10}, params{11}, params{12}, ... bc, bi, bf, bo
                    'forwardDir', obj.forwardDir, ...
                    'scalePredecessor', obj.scalePredecessor);
                obj.aux{1} = outputs{1};
            elseif strcmpi(obj.style, 'coupled')
                [outputs{1}, obj.aux{2}, obj.aux{3}, obj.aux{4}, obj.aux{5}] = ...
                    vl_nnlstmcoupledforward(inputs{1}, inputs{2}, inputs{3}, ... x, h0, c0
                    params{1}, params{2}, params{3}, ... Wc, Wf, Wo
                    params{4}, params{5}, params{6}, ... Uc, Uf, Uo
                    params{7}, params{8}, params{9}, ... bc, bf, bo
                    'forwardDir', obj.forwardDir, ...
                    'scalePredecessor', obj.scalePredecessor);
                obj.aux{1} = outputs{1};
            elseif strcmpi(obj.style, 'peephole')
                [outputs{1}, obj.aux{2}, obj.aux{3}, obj.aux{4}, obj.aux{5}, obj.aux{6}] = ...
                    vl_nnlstmpeepholeforward(inputs{1}, inputs{2}, inputs{3}, ... x, h0, c0
                    params{1}, params{2}, params{3}, params{4}, ... Wc, Wi, Wf, Wo
                    params{5}, params{6}, params{7}, params{8}, ... Uc, Ui, Uf, Uo
                    params{9}, params{10}, params{11}, ... Vi, Vf, Vo
                    params{12}, params{13}, params{14}, params{15}, ... bc, bi, bf, bo
                    'forwardDir', obj.forwardDir, ...
                    'scalePredecessor', obj.scalePredecessor);
                obj.aux{1} = outputs{1};
            elseif strcmpi(obj.style, 'gru')
                [outputs{1}, obj.aux{2}, obj.aux{3}, obj.aux{4}] = ...
                    vl_nngruforward(inputs{1}, inputs{2}, ... x, h0
                    params{1}, params{2}, params{3}, ... Wz, Wr, Wa
                    params{4}, params{5}, params{6}, ... Uz, Ur, Ua
                    params{7}, params{8}, params{9}, ... bz, br, ba
                    'forwardDir', obj.forwardDir);
                obj.aux{1} = outputs{1};
            end
            if useGpu
                outputs = cellfun(@(x) gpuArray(x), outputs, 'UniformOutput', false);
            end
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            useGpu = false;
            if isa(inputs{1}, 'gpuArray')
                useGpu = true;
                inputs = cellfun(@(x) gather(x), inputs, 'UniformOutput', false);
                params = cellfun(@(x) gather(x), params, 'UniformOutput', false);
                derOutputs = cellfun(@(x) gather(x), derOutputs, 'UniformOutput', false);
            end
            if strcmpi(obj.style, 'normal')
                [derInputs{1}, derParams{1}, derParams{2}, derParams{3}, derParams{4}, ...
                    derParams{5}, derParams{6}, derParams{7}, derParams{8}, ...
                    derParams{9}, derParams{10}, derParams{11}, derParams{12}] = ...
                    vl_nnlstmbackward(inputs{1}, inputs{2}, inputs{3}, ...
                    params{1}, params{2}, params{3}, params{4}, ...
                    params{5}, params{6}, params{7}, params{8}, ...
                    params{9}, params{10}, params{11}, params{12}, ...
                    obj.aux{1}, obj.aux{2}, ... h, c
                    obj.aux{3}, obj.aux{4}, obj.aux{5}, obj.aux{6}, derOutputs{1}, ... a, i, f, o, dy
                    'forwardDir', obj.forwardDir, ...
                    'scalePredecessor', obj.scalePredecessor);
                derInputs{2} = [];
                derInputs{3} = [];
            elseif strcmpi(obj.style, 'coupled')
                [derInputs{1}, derParams{1}, derParams{2}, derParams{3}, ...
                    derParams{4}, derParams{5}, derParams{6}, ...
                    derParams{7}, derParams{8}, derParams{9}] = ...
                    vl_nnlstmcoupledbackward(inputs{1}, inputs{2}, inputs{3}, ...
                    params{1}, params{2}, params{3}, ...
                    params{4}, params{5}, params{6}, ...
                    params{7}, params{8}, params{9}, ...
                    obj.aux{1}, obj.aux{2}, ... h, c
                    obj.aux{3}, obj.aux{4}, obj.aux{5}, derOutputs{1}, ... a, f, o, dy
                    'forwardDir', obj.forwardDir, ...
                    'scalePredecessor', obj.scalePredecessor);
                derInputs{2} = [];
                derInputs{3} = [];
            elseif strcmpi(obj.style, 'peephole')
                [derInputs{1}, derParams{1}, derParams{2}, derParams{3}, derParams{4}, ...
                    derParams{5}, derParams{6}, derParams{7}, derParams{8}, ...
                    derParams{9}, derParams{10}, derParams{11}, ...
                    derParams{12}, derParams{13}, derParams{14}, derParams{15}] = ...
                    vl_nnlstmpeepholebackward(inputs{1}, inputs{2}, inputs{3}, ...
                    params{1}, params{2}, params{3}, params{4}, ...
                    params{5}, params{6}, params{7}, params{8}, ...
                    params{9}, params{10}, params{11}, ...
                    params{12}, params{13}, params{14}, params{15}, ...
                    obj.aux{1}, obj.aux{2}, ... h, c
                    obj.aux{3}, obj.aux{4}, obj.aux{5}, obj.aux{6}, derOutputs{1}, ... a, i, f, o, dy
                    'forwardDir', obj.forwardDir, ...
                    'scalePredecessor', obj.scalePredecessor);
                derInputs{2} = [];
                derInputs{3} = [];
            elseif strcmpi(obj.style, 'gru')
                [derInputs{1}, derParams{1}, derParams{2}, derParams{3}, ...
                    derParams{4}, derParams{5}, derParams{6}, ...
                    derParams{7}, derParams{8}, derParams{9}] = ...
                    vl_nngrubackward(inputs{1}, inputs{2}, ...
                    params{1}, params{2}, params{3}, ...
                    params{4}, params{5}, params{6}, ...
                    params{7}, params{8}, params{9}, ...
                    obj.aux{1}, obj.aux{2}, obj.aux{3}, obj.aux{4}, derOutputs{1}, ... h, z, r, a, dy
                    'forwardDir', obj.forwardDir);
                derInputs{2} = [];
            end
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
            dimx = obj.inputDim;
            dimh = obj.hiddenDim;
            dtype = obj.dataType;
            % Xavier improved
            if strcmpi(obj.style, 'normal')
                scW = sqrt(2 / (dimh*dimx));
                scU = sqrt(2 / (dimh*dimh));
                params{ 1} = randn([dimh dimx], dtype) * scW;
                params{ 2} = randn([dimh dimx], dtype) * scW;
                params{ 3} = randn([dimh dimx], dtype) * scW;
                params{ 4} = randn([dimh dimx], dtype) * scW;
                params{ 5} = randn([dimh dimh], dtype) * scU;
                params{ 6} = randn([dimh dimh], dtype) * scU;
                params{ 7} = randn([dimh dimh], dtype) * scU;
                params{ 8} = randn([dimh dimh], dtype) * scU;
                params{ 9} = zeros([dimh 1], dtype);
                params{10} = zeros([dimh 1], dtype);
                params{11} = zeros([dimh 1], dtype);
                params{12} = zeros([dimh 1], dtype);
            elseif strcmpi(obj.style, 'coupled')
                scW = sqrt(2 / (dimh*dimx));
                scU = sqrt(2 / (dimh*dimh));
                params{1} = randn([dimh dimx], dtype) * scW;
                params{2} = randn([dimh dimx], dtype) * scW;
                params{3} = randn([dimh dimx], dtype) * scW;
                params{4} = randn([dimh dimh], dtype) * scU;
                params{5} = randn([dimh dimh], dtype) * scU;
                params{6} = randn([dimh dimh], dtype) * scU;
                params{7} = zeros([dimh 1], dtype);
                params{8} = zeros([dimh 1], dtype);
                params{9} = zeros([dimh 1], dtype);
            elseif strcmpi(obj.style, 'peephole')
                scW = sqrt(2 / (dimh*dimx));
                scU = sqrt(2 / (dimh*dimh));
                scV = sqrt(2 / (dimh*dimh));
                params{ 1} = randn([dimh dimx], dtype) * scW;
                params{ 2} = randn([dimh dimx], dtype) * scW;
                params{ 3} = randn([dimh dimx], dtype) * scW;
                params{ 4} = randn([dimh dimx], dtype) * scW;
                params{ 5} = randn([dimh dimh], dtype) * scU;
                params{ 6} = randn([dimh dimh], dtype) * scU;
                params{ 7} = randn([dimh dimh], dtype) * scU;
                params{ 8} = randn([dimh dimh], dtype) * scU;
                params{ 9} = randn([dimh dimh], dtype) * scV;
                params{10} = randn([dimh dimh], dtype) * scV;
                params{11} = randn([dimh dimh], dtype) * scV;
                params{12} = zeros([dimh 1], dtype);
                params{13} = zeros([dimh 1], dtype);
                params{14} = zeros([dimh 1], dtype);
                params{15} = zeros([dimh 1], dtype);
            elseif strcmpi(obj.style, 'gru')
                scW = sqrt(2 / (dimh*dimx));
                scU = sqrt(2 / (dimh*dimh));
                params{1} = randn([dimh dimx], dtype) * scW;
                params{2} = randn([dimh dimx], dtype) * scW;
                params{3} = randn([dimh dimx], dtype) * scW;
                params{4} = randn([dimh dimh], dtype) * scU;
                params{5} = randn([dimh dimh], dtype) * scU;
                params{6} = randn([dimh dimh], dtype) * scU;
                params{7} = zeros([dimh 1], dtype);
                params{8} = zeros([dimh 1], dtype);
                params{9} = zeros([dimh 1], dtype);
            end
        end
        
        function obj = LSTM2D(varargin)
            obj.load(varargin);
        end
    end
end
