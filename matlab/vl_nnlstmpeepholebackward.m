function [dx, dWc, dWi, dWf, dWo, dUc, dUi, dUf, dUo, dVi, dVf, dVo, dbc, dbi, dbf, dbo] = ...
    vl_nnlstmpeepholebackward(x, h0, c0, Wc, Wi, Wf, Wo, Uc, Ui, Uf, Uo, Vi, Vf, Vo, bc, bi, bf, bo, ...
    h, c, a, i, f, o, dy, varargin)
% VL_LSTMPEEPHOLEBACKWARD Backward pass for vanilla 2D-LSTM
% Syntax:
%   [dx, dWc, dWi, dWf, dWo, dUc, dUi, dUf, dUo, dVi, dVf, dVo, dbc, dbi, dbf, dbo] = ...
%       vl_lstmpeepholebackward(x, h0, c0, Wc, Wi, Wf, Wo, Uc, Ui, Uf, Uo, Vi, Vf, Vo, bc, bi, bf, bo, ...
%       h, c, a, i, f, o, dy, varargin)
%
% Equations:
%   a(t) = tanh(Wc * x(t) + Uc * h(t-1) + bc)
%   i(t) = sigm(Wi * x(t) + Ui * h(t-1) + bi + Vi * c(t-1))
%   f(t) = sigm(Wf * x(t) + Uf * h(t-1) + bf + Vf * c(t-1))
%   o(t) = sigm(Wo * x(t) + Uo * h(t-1) + bo + Vo * c(t))
%   c(t) = i(t) .* a(t) + f(t) .* c(t-1)
%   h(t) = o(t) .* tanh(c(t))

opts.forwardDir = [1 1];
opts.scalePredecessor = true;
opts = vl_argparse(opts, varargin, 'nonrecursive');

[dimx, dimh] = deal(size(x,3), size(Uc,1));
batchSize = size(x, 4);
[T1, T2] = deal(size(x,1), size(x,2));

if isempty(h)
    [h, c, a, i, f, o] = vl_nnlstmpeepholeforward(x, h0, c0, ...
        Wc, Wi, Wf, Wo, Uc, Ui, Uf, Uo, Vi, Vf, Vo, bc, bi, bf, bo, varargin{:});
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
[dh, dc] = deal(zeros([dimh 1 T1 T2 batchSize], 'like', h));
dx = zeros([dimx 1 T1 T2 batchSize], 'like', x);
[dWc, dWi, dWf, dWo] = deal(zeros([dimh dimx 1 1 batchSize], 'like', Wc));
[dUc, dUi, dUf, dUo] = deal(zeros([dimh dimh 1 1 batchSize], 'like', Uc));
[dVi, dVf, dVo] = deal(zeros([dimh dimh 1 1 batchSize], 'like', Vi));
[dbc, dbi, dbf, dbo] = deal(zeros([dimh 1 1 1 batchSize], 'like', h));
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
        dh(:,1,t1,t2,:) = dy(:,1,t1,t2,:);
        if t1 ~= startIdx1
            dh(:,1,t1,t2,:) = dh(:,1,t1,t2,:) + dh(:,1,t1-step1,t2,:);
        end
        if t2 ~= startIdx2
            dh(:,1,t1,t2,:) = dh(:,1,t1,t2,:) + dh(:,1,t1,t2-step2,:);
        end
        tanhc = tanh(c(:,1,t1,t2,:));
        do = dh(:,1,t1,t2,:).*tanhc;
        do = do.*o(:,1,t1,t2,:).*(1-o(:,1,t1,t2,:));
        dc(:,1,t1,t2,:) = dh(:,1,t1,t2,:).*o(:,1,t1,t2,:).*(1-tanhc.^2) + ...
            mtimesx(Vo, 'T', do);
        if t1 ~= startIdx1
            dc(:,1,t1,t2,:) = dc(:,1,t1,t2,:) + dc(:,1,t1-step1,t2,:);
        end
        if t2 ~= startIdx2
            dc(:,1,t1,t2,:) = dc(:,1,t1,t2,:) + dc(:,1,t1,t2-step2,:);
        end
        da = dc(:,1,t1,t2,:).*i(:,1,t1,t2,:);
        di = dc(:,1,t1,t2,:).*a(:,1,t1,t2,:);
        hsucc = zero;
        csucc = zero;
        nsucc = 0;
        if t1 == endIdx1 && t2 == endIdx2
            hsucc = bsxfun(@plus, hsucc, h0);
            csucc = bsxfun(@plus, csucc, c0);
            nsucc = nsucc + 1;
        end
        if t1 ~= endIdx1
            hsucc = hsucc + h(:,1,t1+step1,t2,:);
            csucc = csucc + c(:,1,t1+step1,t2,:);
            nsucc = nsucc + 1;
        end
        if t2 ~= endIdx2
            hsucc = hsucc + h(:,1,t1,t2+step2,:);
            csucc = csucc + c(:,1,t1,t2+step2,:);
            nsucc = nsucc + 1;
        end
        if ~opts.scalePredecessor, nsucc = 1; end
        hsucc = hsucc/nsucc;
        csucc = csucc/nsucc;
        df = dc(:,1,t1,t2,:).*csucc;
        da = da.*(1-a(:,1,t1,t2,:).^2);
        di = di.*i(:,1,t1,t2,:).*(1-i(:,1,t1,t2,:));
        df = df.*f(:,1,t1,t2,:).*(1-f(:,1,t1,t2,:));
        dc(:,1,t1,t2,:) = dc(:,1,t1,t2,:).*f(:,1,t1,t2,:) + ...
            mtimesx(Vf, 'T', df) + mtimesx(Vi, 'T', di);
        dh(:,1,t1,t2,:) = mtimesx(Uc, 'T', da) + mtimesx(Ui, 'T', di) + ...
            mtimesx(Uf, 'T', df) + mtimesx(Uo, 'T', do);
        dc(:,1,t1,t2,:) = dc(:,1,t1,t2,:)/nsucc;
        dh(:,1,t1,t2,:) = dh(:,1,t1,t2,:)/nsucc;
        dbc = dbc + da;
        dbi = dbi + di;
        dbf = dbf + df;
        dbo = dbo + do;
        dWc = dWc + mtimesx(da, x(:,1,t1,t2,:), 'T');
        dWi = dWi + mtimesx(di, x(:,1,t1,t2,:), 'T');
        dWf = dWf + mtimesx(df, x(:,1,t1,t2,:), 'T');
        dWo = dWo + mtimesx(do, x(:,1,t1,t2,:), 'T');
        dUc = dUc + mtimesx(da, hsucc, 'T');
        dUi = dUi + mtimesx(di, hsucc, 'T');
        dUf = dUf + mtimesx(df, hsucc, 'T');
        dUo = dUo + mtimesx(do, hsucc, 'T');
        dVi = dVi + mtimesx(di, csucc, 'T');
        dVf = dVf + mtimesx(df, csucc, 'T');
        dVo = dVo + mtimesx(do, c(:,1,t1,t2,:), 'T');
        dx(:,1,t1,t2,:) = mtimesx(Wc, 'T', da) + mtimesx(Wi, 'T', di) + ...
            mtimesx(Wf, 'T', df) + mtimesx(Wo, 'T', do);
    end
end

%% Output
% Rearrange dimensions of dx
dx = reshape(dx, [dimx T1 T2 batchSize]);
dx = permute(dx, [2 3 1 4]);

% Sum over batch
dWc = sum(dWc, 5);
dWi = sum(dWi, 5);
dWf = sum(dWf, 5);
dWo = sum(dWo, 5);
dUc = sum(dUc, 5);
dUi = sum(dUi, 5);
dUf = sum(dUf, 5);
dUo = sum(dUo, 5);
dVi = sum(dVi, 5);
dVf = sum(dVf, 5);
dVo = sum(dVo, 5);
dbc = sum(dbc, 5);
dbi = sum(dbi, 5);
dbf = sum(dbf, 5);
dbo = sum(dbo, 5);
