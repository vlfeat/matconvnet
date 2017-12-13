function [dx, dWz, dWr, dWa, dUz, dUr, dUa, dbz, dbr, dba] = ...
    vl_nngrubackward(x, h0, Wz, Wr, Wa, Uz, Ur, Ua, bz, br, ba, h, z, r, a, dy, varargin)
% VL_LSTMGRUBACKWARD Backward pass for 2D-GRU
% Syntax:
%   [dx, dWc, dWf, dWo, dUc, dUf, dUo, dbc, dbf, dbo] = ...
%       vl_grubackward(x, h0, Wz, Wr, Wa, Uz, Ur, Ua, bz, br, ba, h, z, r, a, dy, varargin)
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

if isempty(h)
    [h, z, r, a] = vl_nngruforward(x, h0, Wz, Wr, Wa, Uz, Ur, Ua, ...
        bz, br, ba, varargin{:});
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
dh = zeros([dimh 1 T1 T2 batchSize], 'like', h);
dx = zeros([dimx 1 T1 T2 batchSize], 'like', x);
[dWz, dWr, dWa] = deal(zeros([dimh dimx 1 1 batchSize], 'like', Wz));
[dUz, dUr, dUa] = deal(zeros([dimh dimh 1 1 batchSize], 'like', Uz));
[dbz, dbr, dba] = deal(zeros([dimh 1 1 1 batchSize], 'like', h));
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
        hsucc = hsucc/nsucc;
        da = dh(:,1,t1,t2,:).*z(:,1,t1,t2,:);
        dz = dh(:,1,t1,t2,:).*(a(:,1,t1,t2,:)-hsucc);
        da = da.*(1-a(:,1,t1,t2,:).^2);
        dz = dz.*(z(:,1,t1,t2,:).*(1-z(:,1,t1,t2,:)));
        dhr = mtimesx(Ua, 'T', da);
        dr = dhr.*hsucc;
        dr = dr.*(r(:,1,t1,t2,:).*(1-r(:,1,t1,t2,:)));
        dh(:,1,t1,t2,:) = dh(:,1,t1,t2,:).*(1-z(:,1,t1,t2,:)) + ...
            mtimesx(Uz, 'T', dz) + mtimesx(Ur, 'T', dr) + dhr.*r(:,1,t1,t2,:);
        dh(:,1,t1,t2,:) = dh(:,1,t1,t2,:)/nsucc;
        dbz = dbz + dz;
        dbr = dbr + dr;
        dba = dba + da;
        dWz = dWz + mtimesx(dz, x(:,1,t1,t2,:), 'T');
        dWr = dWr + mtimesx(dr, x(:,1,t1,t2,:), 'T');
        dWa = dWa + mtimesx(da, x(:,1,t1,t2,:), 'T');
        dUz = dUz + mtimesx(dz, hsucc, 'T');
        dUr = dUr + mtimesx(dr, hsucc, 'T');
        dUa = dUa + mtimesx(da, hsucc.*r(:,1,t1,t2,:), 'T');
        dx(:,1,t1,t2,:) = mtimesx(Wa, 'T', da) + ...
            mtimesx(Wz, 'T', dz) + mtimesx(Wr, 'T', dr);
    end
end

%% Output
% Rearrange dimensions of dx
dx = reshape(dx, [dimx T1 T2 batchSize]);
dx = permute(dx, [2 3 1 4]);

% Sum over batch
dWz = sum(dWz, 5);
dWr = sum(dWr, 5);
dWa = sum(dWa, 5);
dUz = sum(dUz, 5);
dUr = sum(dUr, 5);
dUa = sum(dUa, 5);
dbz = sum(dbz, 5);
dbr = sum(dbr, 5);
dba = sum(dba, 5);
