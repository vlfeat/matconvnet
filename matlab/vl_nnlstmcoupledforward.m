function [h, c, a, f, o] = vl_nnlstmcoupledforward(x, h0, c0, ...
    Wc, Wf, Wo, Uc, Uf, Uo, bc, bf, bo, varargin)
% VL_LSTMCOUPLEDFORWARD Forward pass for coupled 2D-LSTM
% Syntax:
%   [h, c, a, f, o] = vl_lstmcoupledforward(x, h0, c0, ...
%       Wc, Wf, Wo, Uc, Uf, Uo, bc, bf, bo, varargin)
%
% Equations:
%   a(t) = tanh(Wc * x(t) + Uc * h(t-1) + bc)
%   f(t) = sigm(Wf * x(t) + Uf * h(t-1) + bf)
%   o(t) = sigm(Wo * x(t) + Uo * h(t-1) + bo)
%   c(t) = f(t) .* c(t-1) + (1-f(t)) .* a(t)
%   h(t) = o(t) .* tanh(c(t))

opts.forwardDir = [1 1];
opts.scalePredecessor = true;
opts = vl_argparse(opts, varargin, 'nonrecursive');

[dimx, dimh] = deal(size(x,3), size(Uc,1));
batchSize = size(x, 4);
[T1, T2] = deal(size(x,1), size(x,2));

% Rearrange dimensions of x
x = permute(x, [3 1 2 4]);
x = reshape(x, [dimx 1 T1 T2 batchSize]);

% Outputs and Intermediate results
[h, c, a, f, o] = deal(zeros([dimh 1 T1 T2 batchSize], 'like', x));
zero = zeros([dimh 1 1 1 batchSize], 'like', x);

%% Forward pass
if opts.forwardDir(1) > 0
    [step1, startIdx1, endIdx1] = deal(1, 1, T1);
else
    [step1, startIdx1, endIdx1] = deal(-1, T1, 1);
end
if opts.forwardDir(2) > 0
    [step2, startIdx2, endIdx2] = deal(1, 1, T2);
else
    [step2, startIdx2, endIdx2] = deal(-1, T2, 1);
end
for t2 = startIdx2:step2:endIdx2
    for t1 = startIdx1:step1:endIdx1
        hpred = zero;
        cpred = zero;
        npred = 0;
        if t1 == startIdx1 && t2 == startIdx2
            hpred = bsxfun(@plus, hpred, h0);
            cpred = bsxfun(@plus, cpred, c0);
            npred = npred + 1;
        end
        if t1 ~= startIdx1
            hpred = hpred + h(:,1,t1-step1,t2,:);
            cpred = cpred + c(:,1,t1-step1,t2,:);
            npred = npred + 1;
        end
        if t2 ~= startIdx2
            hpred = hpred + h(:,1,t1,t2-step2,:);
            cpred = cpred + c(:,1,t1,t2-step2,:);
            npred = npred + 1;
        end
        if ~opts.scalePredecessor, npred = 1; end
        hpred = hpred/npred;
        cpred = cpred/npred;
        a(:,1,t1,t2,:) =  bsxfun(@plus, mtimesx(Wc,x(:,1,t1,t2,:)) + mtimesx(Uc,hpred), bc);
        f(:,1,t1,t2,:) =  bsxfun(@plus, mtimesx(Wf,x(:,1,t1,t2,:)) + mtimesx(Uf,hpred), bf);
        o(:,1,t1,t2,:) =  bsxfun(@plus, mtimesx(Wo,x(:,1,t1,t2,:)) + mtimesx(Uo,hpred), bo);
        a(:,1,t1,t2,:) = tanh(a(:,1,t1,t2,:));
        f(:,1,t1,t2,:) = sigm(f(:,1,t1,t2,:));
        o(:,1,t1,t2,:) = sigm(o(:,1,t1,t2,:));
        c(:,1,t1,t2,:) = (1-f(:,1,t1,t2,:)).*a(:,1,t1,t2,:) + f(:,1,t1,t2,:).*cpred;
        h(:,1,t1,t2,:) = o(:,1,t1,t2,:) .* tanh(c(:,1,t1,t2,:));
    end
end

%%
% Rearrange dimensions of h
h = reshape(h, [dimh T1 T2 batchSize]);
h = permute(h, [2 3 1 4]);

%% Transfer function
function y = sigm(x)
y = 1./(1+exp(-x));
