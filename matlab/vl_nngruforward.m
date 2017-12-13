function [h, z, r, a] = vl_nngruforward(x, h0, Wz, Wr, Wa, Uz, Ur, Ua, bz, br, ba, varargin)
% VL_GRUFORWARD Forward pass for 2D-GRU
% Syntax:
%   [h, z, r, a] = vl_gruforward(x, h0, Wz, Wr, Wa, Uz, Ur, Ua, bz, br, ba, varargin)
%
% Equations:
%   z(t) = sigm(Wz * x(t) + Uz * h(t-1) + bz)
%   r(t) = sigm(Wr * x(t) + Ur * h(t-1) + br)
%   a(t) = tanh(Wa * x(t) + Ua * (r(t) .* h(t-1)))
%   h(t) = (1-z(t)) .* h(t-1) + z(t) .* a(t)

opts.forwardDir = [1 1];
opts.scalePredecessor = true;
opts = vl_argparse(opts, varargin, 'nonrecursive');

[dimx, dimh] = deal(size(x,3), size(Uz,1));
batchSize = size(x, 4);
[T1, T2] = deal(size(x,1), size(x,2));

% Rearrange dimensions of x
x = permute(x, [3 1 2 4]);
x = reshape(x, [dimx 1 T1 T2 batchSize]);

% Outputs and Intermediate results
[h, z, r, a] = deal(zeros([dimh 1 T1 T2 batchSize], 'like', x));
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
        hpred = hpred/npred;
        z(:,1,t1,t2,:) = sigm(bsxfun(@plus, mtimesx(Wz,x(:,1,t1,t2,:)) + ...
            mtimesx(Uz,hpred), bz));
        r(:,1,t1,t2,:) = sigm(bsxfun(@plus, mtimesx(Wr,x(:,1,t1,t2,:)) + ...
            mtimesx(Ur,hpred), br));
        a(:,1,t1,t2,:) = tanh(bsxfun(@plus, mtimesx(Wa,x(:,1,t1,t2,:)) + ...
            mtimesx(Ua,(r(:,1,t1,t2,:).*hpred)), ba));
        h(:,1,t1,t2,:) = z(:,1,t1,t2,:).*a(:,1,t1,t2,:) + ...
            (1-z(:,1,t1,t2,:)).*hpred;
    end
end

%%
% Rearrange dimensions of h
h = reshape(h, [dimh T1 T2 batchSize]);
h = permute(h, [2 3 1 4]);

%% Transfer function
function y = sigm(x)
y = 1./(1+exp(-x));
