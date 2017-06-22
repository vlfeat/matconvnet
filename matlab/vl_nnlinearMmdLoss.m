function[Y] = vl_nnlinearMmdLoss(X, labels, lopts, dzdy)
% -------------------------------------------------------------------------
%VL_NNLINEARMMDLOSS - Linear MMD based loss for domain adaptation
% This function measures the Maximum Mean Discrepancy between the source
% and the target representations in a bacth of data. The data-batch is
% designed to have nearly equal number of source and target data points.
% The linear MMD is based on the paper: 
%  @inproceedings{gretton2012optimal,
%    title={Optimal kernel choice for large-scale two-sample tests},
%    author={Gretton, Arthur and Sejdinovic, Dino and Strathmann, 
%    Heiko and Balakrishnan, Sivaraman and Pontil, 
%    Massimiliano and Fukumizu, 
%    Kenji and Sriperumbudur, Bharath K},
%    booktitle={Advances in neural information processing systems},
%    pages={1205--1213},
%    year={2012}
%  }
%
% Copyright (C) 2016-17 Hemanth Venkateswara.
% All rights reserved.

if lopts.K ~= 0
   % Calculate mmdLoss only if source and target data combined
    checkLabels = reshape(repmat(lopts.C, lopts.K, 1), 1, []);
    if length(labels) > length(checkLabels) && isequal(labels(1:length(checkLabels)), checkLabels)
        mmdLoss = true;
    else
        mmdLoss = false;
    end 
else
    mmdLoss = true;
end

gamma = lopts.gamma;

if gamma == 0
    mmdLoss = false;
end

if ~mmdLoss
    if nargin <= 3 || isempty(dzdy)
        fprintf('Lm = %.03f ', 0);
        Y = X;
        return;
    else
        Y = gpuArray(dzdy);
        return;
    end
end

[H,W,C,N] = size(X);
X = reshape(X, [H*W*C, N]);
d = size(X,1);
srcIds = labels > 0;
tgtIds = labels == 0;
ns = sum(srcIds);
nt = sum(tgtIds);

% assert(ns == nt, 'Number of Source and Target differ in batch');
n = ns + nt;
ns1 = ceil(ns/2); ns2 = ns - ns1; nt1 = ceil(nt/2); nt2 = nt - nt1;
sIdx = find(srcIds); tIdx = find(tgtIds);
sIdx = sIdx(randperm(length(sIdx)));
tIdx = tIdx(randperm(length(tIdx)));
sIdx1 = sIdx(1:ns1); sIdx2 = sIdx(ns1+1:end);
tIdx1 = tIdx(1:nt1); tIdx2 = tIdx(nt1+1:end);

Us1 = X(:, sIdx1);
Us2 = X(:, sIdx2);
Ut1 = X(:, tIdx1);
Ut2 = X(:, tIdx2);

if ns1 > ns2
    isuneven = true;
    Us2(:,end+1) = Us2(:,end);
    Ut2(:,end+1) = Ut2(:,end);
    ns2 = ns2 + 1;
    nt2 = nt2 + 1;
else
    isuneven = false;
end

K1 = zeros(ns1, 1);
K2 = zeros(ns1, 1);
K3 = zeros(ns1, 1);
K4 = zeros(ns1, 1);
Kd1 = sum(Us1.^2)' + sum(Us2.^2)' -2.*sum(Us1.*Us2)';
Kd2 = sum(Us1.^2)' + sum(Ut2.^2)' -2.*sum(Us1.*Ut2)';
Kd3 = sum(Us2.^2)' + sum(Ut1.^2)' -2.*sum(Us2.*Ut1)';
Kd4 = sum(Ut1.^2)' + sum(Ut2.^2)' -2.*sum(Ut1.*Ut2)';
dm2 = median([Kd1(:);Kd2(:);Kd3(:);Kd4(:)]);

Q = 2.^(-8:8);
for qq = Q
  K1 = K1 + exp(-(qq/dm2).*Kd1);
  K2 = K2 + exp(-(qq/dm2).*Kd2);
  K3 = K3 + exp(-(qq/dm2).*Kd3);
  K4 = K4 + exp(-(qq/dm2).*Kd4);
end
lQ = length(Q);
cost = (gamma/(lQ)).*(sum(K1(:)) - sum(K2(:)) - sum(K3(:)) + sum(K4(:)));

if nargin <= 3 || isempty(dzdy)
    fprintf('Lm = %.03f ', cost);
    Y = reshape(X, [H, W, C, N]);
else
    gpuMode = isa(X, 'gpuArray') ;
    gradUs1 = zeros(d, ns1);
    gradUs2 = zeros(d, ns2);
    gradUt1 = zeros(d, nt1);
    gradUt2 = zeros(d, nt2);
    if gpuMode
        Y = gpuArray(single(zeros(d, n)));
    else
        Y = single(zeros(d, n));
    end
    for qq = Q
        qq_diff = -2*qq/dm2;
        % For K1 .. [Us1, Us2]
        Kd1_diff = exp(-(qq/dm2).*Kd1).*qq_diff;
        kd1_diff_grad = bsxfun(@times, Us1 - Us2, Kd1_diff');
        gradUs1 = gradUs1 + kd1_diff_grad;
        gradUs2 = gradUs2 - kd1_diff_grad;
        
        % For K2 .. [Us1, Ut2]
        Kd2_diff = exp(-(qq/dm2).*Kd2).*qq_diff;
        Kd2_diff_grad = bsxfun(@times, Us1 - Ut2, Kd2_diff');
        gradUs1 = gradUs1 - Kd2_diff_grad; % since -K2
        gradUt2 = gradUt2 + Kd2_diff_grad;
        
        % For K3 .. [Us2, Ut1]
        Kd3_diff = exp(-(qq/dm2).*Kd3).*qq_diff;
        Kd3_diff_grad = bsxfun(@times, Us2 - Ut1, Kd3_diff');
        gradUs2 = gradUs2 - Kd3_diff_grad; % since -K3
        gradUt1 = gradUt1 + Kd3_diff_grad;
        
        % For K4 .. [Ut1, Ut2]
        Kd4_diff = exp(-(qq/dm2).*Kd4).*qq_diff;
        Kd4_diff_grad = bsxfun(@times, Ut1 - Ut2, Kd4_diff');
        gradUt1 = gradUt1 + Kd4_diff_grad;
        gradUt2 = gradUt2 - Kd4_diff_grad;
    end
    if isuneven
        gradUs2(:, end) = [];
        gradUt2(:, end) = [];
    end
    Y(:, sIdx1) = gradUs1;
    Y(:, sIdx2) = gradUs2;
    Y(:, tIdx2) = gradUt2;
    Y(:, tIdx1) = gradUt1;
    Y = 1/(lQ).*Y;
    Y = dzdy + (gamma.*reshape(Y, [H, W, C, N]));
end