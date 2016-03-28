classdef PDist < dagnn.ElementWise
    properties
        p = 2
        noRoot = false ;
        epsilon = 1e-6 ;
        aggregate = false ;
        opts = {}
    end
    
    properties (Transient)
        average = 0
        numAveraged = 0
    end
    
    methods
        function obj = PDist(varargin)
            obj.load(varargin) ;
            % normalize field by implicitly calling setters defined in
            % dagnn.Filter and here
            obj.p = obj.p ;
            obj.noRoot = obj.noRoot ;
            obj.epsilon = obj.epsilon ;
            obj.aggregate = obj.aggregate ;
        end
        
        function outputs = forward(obj, inputs, params)
            outputs{1} = vl_nnpdist(inputs{1}, inputs{2}, obj.p, 'noRoot', obj.noRoot, 'epsilon', obj.epsilon, 'aggregate', obj.aggregate, obj.opts{:}) ;
            n = obj.numAveraged ;
            m = n + size(inputs{1},4) ;
            obj.average = (n * obj.average + gather(outputs{1})) / m ;
            obj.numAveraged = m ;
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            derInputs{1} = vl_nnpdist(inputs{1}, inputs{2}, obj.p, derOutputs{1}, 'noRoot', obj.noRoot, 'epsilon', obj.epsilon, 'aggregate', obj.aggregate, obj.opts{:}) ;
            derInputs{2} = [] ;
            derParams = {} ;
        end
        
        function reset(obj)
            obj.average = 0 ;
            obj.numAveraged = 0 ;
        end
        
        function outputSizes = getOutputSizes(obj, inputSizes, paramSizes)
            outputSizes{1} = [1 1 1 inputSizes{1}(4)] ;
        end
        
        function rfs = getReceptiveFields(obj)
            % the receptive field depends on the dimension of the variables
            % which is not known until the network is run
            rfs(1,1).size = [NaN NaN] ;
            rfs(1,1).stride = [NaN NaN] ;
            rfs(1,1).offset = [NaN NaN] ;
            rfs(2,1) = rfs(1,1) ;
        end
        
        function obj = Loss(varargin)
            obj.load(varargin) ;
        end
    end
end
