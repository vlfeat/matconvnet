function [dx, dh0, dU, dW, db] = vl_nnrnnbackward(x, h0, U, W, b, h, dy, varargin)
% VL_RNNBACKWARD Backward pass for vanilla 2D-RNN
% Syntax:
%   [dx, dh0, dU, dW, db] = vl_rnnbackward(x, h0, U, W, b, [], dy, varargin),
%   when called by RNN2D
%   [dx, dh0, dU, dW, db] = vl_rnnbackward(x, h0, U, W, [], h, dy, varargin),
%   when called by RNN2DAux
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

if isempty(h)
    [h] = vl_nnrnnforward(x, h0, U, W, b, varargin{:});
end

% Rearrange dimensions of h
h = permute(h, [3 1 2 4]);
h = reshape(h, [dimh 1 T1 T2 batchSize]);

% Rearrange dimensions of x
x = permute(x, [3 1 2 4]);
x = reshape(x, [dimx 1 T1 T2 batchSize]);

% Rearrange dimensions of dy
dy = permute(dy, [3 1 2 4]);
dy = reshape(dy, [dimh 1 T1 T2 batchSize]);

% Outputs and Intermediate results
dU = zeros([dimh dimx 1 1 batchSize], 'like', U);
dW = zeros([dimh dimh 1 1 batchSize], 'like', W);
db = zeros([dimh 1 1 1 batchSize], 'like', dy);
dh = zeros([dimh 1 T1 T2 batchSize], 'like', h);
dx = zeros([dimx 1 T1 T2 batchSize], 'like', x);
zero = zeros([dimh 1 1 1 batchSize], 'like', h);

%% Backward
if opts.forwardDir(1) > 0
    [step1, startIdx1, endIdx1] = deal(-1, T1, 1);
else
    [step1, startIdx1, endIdx1] = deal(1, 1, T1);
end
if opts.forwardDir(2) > 0
    [step2, startIdx2, endIdx2] = deal(-1, T2, 1);
else
    [step2, startIdx2, endIdx2] = deal(1, 1, T2);
end
for t2 = startIdx2:step2:endIdx2
    for t1 = startIdx1:step1:endIdx1
        dh_ = dy(:,1,t1,t2,:);
        if t1 ~= startIdx1
            dh_ = dh_ + dh(:,1,t1-step1,t2,:);
        end
        if t2 ~= startIdx2
            dh_ = dh_ + dh(:,1,t1,t2-step2,:);
        end
        da = opts.hiddenFn(h(:,1,t1,t2,:), dh_);
        db = db + da;
        hsucc = zero;
        nsucc = 0;
        if t1 == endIdx1 && t2 == endIdx2
            hsucc = bsxfun(@plus, hsucc, h0);
            nsucc = nsucc + 1;
        end
        if t1 ~= endIdx1
            hsucc = hsucc + h(:,1,t1+step1,t2,:);
            nsucc = nsucc + 1;
        end
        if t2 ~= endIdx2
            hsucc = hsucc + h(:,1,t1,t2+step2,:);
            nsucc = nsucc + 1;
        end
        if ~opts.scalePredecessor, nsucc = 1; end
        hsucc = hsucc/nsucc;
        dh(:,1,t1,t2,:) = mtimesx(W, 'T', da);
        dh(:,1,t1,t2,:) = dh(:,1,t1,t2,:)/nsucc;
        dW = dW + mtimesx(da, hsucc, 'T');
        dU = dU + mtimesx(da, x(:,1,t1,t2,:), 'T');
        dx(:,1,t1,t2,:) = mtimesx(U, 'T', da);
    end
end
dh0 = mtimesx(W, 'T', da);

%% Output
% Rearrange dimensions of dx
dx = reshape(dx, [dimx T1 T2 batchSize]);
dx = permute(dx, [2 3 1 4]);
% Sum over batch
dU = sum(dU, 5);
dW = sum(dW, 5);
db = sum(db, 5);
dh0 = sum(dh0, 5);
