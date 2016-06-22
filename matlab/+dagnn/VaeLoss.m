classdef VaeLoss < dagnn.Loss
    
    properties (Transient)
        t
    end
    
    methods
        function outputs = forward(obj, inputs, params)
            %       logx=min(max(log(inputs{1}),log(eps)),log(1-eps));
            %       log1_x=min(max(log(1-inputs{1}),log(eps)),log(1-eps));
            %       x= min(max(double(inputs{1}),eps(1)),1-eps(1));
            neps=134217729*eps;
            x= min(max(inputs{1},eps),1-neps);
            xent_loss = -mean(inputs{2}.*log(x)+(1-inputs{2}).*log(1-x),3);
            kl_loss =- 0.5 * mean(1 + inputs{4} - inputs{3}.^2 - exp(inputs{4}), 3);
            obj.t=xent_loss+kl_loss;
            outputs{1} = sum(obj.t(:));
            n = obj.numAveraged ;
            m = n + size(inputs{1},4) ;
            obj.average = (n * obj.average + gather(outputs{1})) / m ;
            obj.numAveraged = m ;
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            %        x= min(max(double(inputs{1}),eps(1)),1-eps(1));
            neps=134217729*eps;
            x= min(max(inputs{1},eps),1-neps);
            derInputs{1} = derOutputs{1}.*(-inputs{2}./x+(1-inputs{2})./(1-x))/size(inputs{1},3);
            derInputs{2} = [] ;
            derInputs{3} = derOutputs{1}.*inputs{3}/size(inputs{3},3) ;
            derInputs{4} =  -.5*derOutputs{1}.*(1-exp(inputs{4}))/size(inputs{4},3) ;
            derParams = {} ;
        end
        
    end
end
