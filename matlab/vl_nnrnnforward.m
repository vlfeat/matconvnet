function [h] = vl_nnrnnforward(x, h0, U, W, b, varargin)
% VL_RNNFORWARD Forward pass for vanilla 2D-RNN
% Syntax:
%   [h] = vl_rnnforward(x, h0, U, W, b, varargin)
%
% Forward equations:
%   a(t) = W*h(t-1) + U*x(t) + b;
%   h(t) = hiddenFn(a(t));
%
% Matrix dimension:
%   U: [dimh dimx]
%   W: [dimh dimh]
%
% Input and output variable dimension:
%    x: [T1 T2 dimx batchSize]
%    h: [dimh 1 T1 T2 batchSize] % for faster slicing
%   h0: [dimh 1]

opts.forwardDir = [1 1];
opts.hiddenFn = @nnfun.tanh;
opts.derFnInput = 'y';
opts.scalePredecessor = true;
opts = vl_argparse(opts, varargin, 'nonrecursive');

[dimx, dimh] = deal(size(x,3), size(U,1));
batchSize = size(x, 4);
[T1, T2] = deal(size(x,1), size(x,2));

% Rearrange dimensions of x
x = permute(x, [3 1 2 4]);
x = reshape(x, [dimx 1 T1 T2 batchSize]);

% Outputs and Intermediate results
h = zeros([dimh 1 T1 T2 batchSize], 'like', x);
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
        npred = 0;
        if t1 == startIdx1 && t2 == startIdx2
            hpred = bsxfun(@plus, hpred, h0);
            npred = npred + 1;
        end
        if t1 ~= startIdx1
            hpred = hpred + h(:,1,t1-step1,t2,:);
            npred = npred + 1;
        end
        if t2 ~= startIdx2
            hpred = hpred + h(:,1,t1,t2-step2,:);
            npred = npred + 1;
        end
        if ~opts.scalePredecessor, npred = 1; end
        hpred = hpred/npred;
        a = bsxfun(@plus, mtimesx(U,x(:,1,t1,t2,:)) + mtimesx(W,hpred), b);
        h(:,1,t1,t2,:) = opts.hiddenFn(a, []);
    end
end

%%
% Rearrange dimensions of h
h = reshape(h, [dimh T1 T2 batchSize]);
h = permute(h, [2 3 1 4]);
