classdef Gaussian < dagnn.ElementWise
    properties
        epsilon_std=1;
    end
    properties(Transient)
        e
    end
    methods
%         function obj = Gaussian(eps_std)
%             %GAUSSIAN constructor
%             %  OBJ = GAUSSIAN(EPS_STD) creates a Gaussian sampling layer
%             %  with initial standard deviation EPS_STD
%             obj.epsilon_std=eps_std;
%         end
        
        function outputs = forward(obj, inputs, params)
            %FORWARD Forward step
            %  OUTPUTS = FORWARD(OBJ, INPUTS, PARAMS) takes the layer object OBJ
            %  and cell arrays of inputs and parameters and produces a cell
            %  array of outputs evaluating the layer forward.
            assert(all(size(inputs{1})==size(inputs{2})),'Size of inputs{1} and inputs{2} must be equal');
            obj.e=randn(size(inputs{1}))*obj.epsilon_std;
            outputs = {inputs{1}+obj.e.*exp(inputs{2})} ;
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            %BACKWARD  Bacwkard step
            %  [DERINPUTS, DERPARAMS] = BACKWARD(OBJ, INPUTS, INPUTS, PARAMS,
            %  DEROUTPUTS) takes the layer object OBJ and cell arrays of
            %  inputs, parameters, and output derivatives and produces cell
            %  arrays of input and parameter derivatives evaluating the layer
            %  backward.
            derInputs{1} = derOutputs{1} ;
            derInputs{2} = derOutputs{1}.*obj.e ;
            derParams = {} ;
        end
        
        function outputSizes = getOutputSizes(obj, inputSizes)
            outputSizes{1} = inputSizes{1} ;
            for k = 2:numel(inputSizes)
                if all(~isnan(inputSizes{k})) && all(~isnan(outputSizes{1}))
                    if ~isequal(inputSizes{k}, outputSizes{1})
                        warning('Gaussian layer: the dimensions of the input variables is not the same.') ;
                    end
                end
            end
        end
        function obj = Gaussian(varargin)
            obj.load(varargin) ;
        end
    end % methods
    
end % classdef