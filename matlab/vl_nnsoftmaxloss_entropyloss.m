function y = vl_nnsoftmaxloss_entropyloss(x,labels,lopts,dzdy)
%VL_NNSOFTMAXLOSS_ENTROPYLOSS combines VL_SOFTMAXLOSS with entropy loss for
%da_hash. This function adds an entropy based loss on top of the
%softmaxloss. These results are not included in the CVPR 2017 paper.
%
%VL_NNSOFTMAXLOSS_ENTROPYLOSS CNN combined softmax and logistic loss 
%with entropy loss.
%
%   Y = VL_NNSOFTMAXLOSS_ENTROPYLOSS(X, LABELS, LOPTS) applies the softmax 
%   operator followed by
%   the logistic loss the data X. X has dimension H x W x hashSize x N,
%   packing N arrays of W x H hashSize-dimensional vectors.
%
%   LABELS contains the class labels, which should be integers in the range
%   0 to C. 0 indicates it is a target data point. 
%   LABELS is an array with N elements.
%   LOPTS has the following fields
%     - `lopts.K` the K parameter for the da_hash
%     - `lopts.C` the labels in the dataset (1,2,...,C)
%     - `lopts.l1` Weight for hash loss
%     - `lopts.entpW` Weight for entropy loss
%
%   DZDX = VL_NNSOFTMAXLOSS_ENTROPYLOSS(X, LABELS, LOPTS, DZDY) computes 
%   the derivative of the block projected onto DZDY. DZDX and DZDY have the 
%   same dimensions as X and Y respectively.

% The entropyloss is the unsupervised entropy loss for unlabeled data

% Copyright (C) 2016-17 Hemanth Venkateswara.
% based on code from Andrea Vedaldi
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

% work around a bug in MATLAB, where native cast() would slow
% progressively
if isa(x, 'gpuArray')
  switch classUnderlying(x) ;
    case 'single', cast = @(z) single(z) ;
    case 'double', cast = @(z) double(z) ;
  end
else
  switch class(x)
    case 'single', cast = @(z) single(z) ;
    case 'double', cast = @(z) double(z) ;
  end
end

checkLabels = reshape(repmat(lopts.C, lopts.K, 1), 1, []);
c = labels;
if length(labels) > length(checkLabels) && isequal(labels(1:length(checkLabels)), checkLabels)
    jeLoss = true;
else
    jeLoss = false;
end
l1 = lopts.l1; % Weight for hash Similarity loss
entpW = lopts.entpW; % Weight for Entropy loss

U = squeeze(x); % convert to D x N array
srcIds = labels > 0;
tgtIds = labels == 0;
ns = sum(srcIds);
nt = sum(tgtIds);
Us = U(:,srcIds);
Usmax = max(Us); % find max of each column
eUs = exp(bsxfun(@minus, Us, Usmax)) ;
c = labels(srcIds);
idx = 0 : numel(c)-1;
cidx = max(lopts.C)*idx + c;

if nargin <= 3 % changed from 2 to include lopts
  y = l1*sum(Usmax + log(sum(eUs)) - reshape(Us(cidx), 1, numel(cidx)));
  if jeLoss
      Lst = entpW.*jointEntropyCostAndGradComplex(U, labels, lopts.K, 1);
      y = y + sum(Lst(:));
  end
else
  gpuMode = isa(U, 'gpuArray') ;
  if gpuMode
      gradst = gpuArray(zeros(size(Us,1), (ns + nt)));
  else
      gradst = zeros(size(Us,1), (ns + nt));
  end
  gradS = bsxfun(@rdivide, eUs, sum(eUs));
  gradS(cidx) = gradS(cidx) - 1;
  gradst(:, srcIds) = l1.*gradS;
  if jeLoss
      gradJE = entpW.*jointEntropyCostAndGradComplex(U, labels, lopts.K, 0);
      gradst = gradst + gradJE;
  end
  y = dzdy.*reshape(gradst, [1,1,size(gradst)]);
end


end

% -------------------------------------------------------------------------
function[Y] = jointEntropyCostAndGradComplex(U, c, K, doCost)
% -------------------------------------------------------------------------
labels = squeeze(c);
srcIds = labels > 0;
tgtIds = labels == 0;
nt = sum(tgtIds);
ns = sum(srcIds);
Ut = U(:,tgtIds);
Us = U(:,srcIds);
srcLabels = labels(srcIds);

unqLabs = unique(srcLabels);
C = length(unqLabs);
d = size(U,1);
gpuMode = isa(U, 'gpuArray') ;

UtUsdot = Ut'*Us; % nt x ns
Utsmax = max(UtUsdot, [], 2);
Pijk = exp(bsxfun(@minus, UtUsdot, Utsmax));
if any(isnan(gather(Pijk)))
    error('Pijk is nan');
end
% Pijk = exp(UtUsdot);
Pijk = bsxfun(@rdivide, Pijk, sum(Pijk,2)); % nt x ns
if gpuMode
    Pij = gpuArray(single(zeros(nt, C)));
else
    Pij = single(zeros(nt, C));
end
onesk = ones(K,1);
for ii = 1:C 
    ids = (ii-1)*K+1 : ii*K;
    Pij(:,ii) = Pijk(:, ids)*onesk;
end

if doCost
    Ph = Pij.*log(Pij);
    Ph(isnan(gather(Ph))) = 0;
    Y = -sum(Ph(:));
else
    % Grad for vi
    if gpuMode
        PijUj = gpuArray(single(zeros(d, nt))); % For sum_k P_ijk*U_jk
        OnePlusLogPijk = gpuArray(single(zeros(nt, ns))); % Repeat Pij and make it [nt x ns]
    else
        PijUj = single(zeros(d, nt)); % For sum_k P_ijk*U_jk
        OnePlusLogPijk = single(zeros(nt, ns)); % Repeat Pij and make it [nt x ns]
    end
    OnePlusLogPij = (1+log(Pij));
    OnePlusLogPij(isinf(OnePlusLogPij)) = 0;
    for ii = 1:C
        ids = (ii-1)*K+1 : ii*K;
        OnePlusLogPijRep = repmat(OnePlusLogPij(:,ii), 1, K);
        OnePlusLogPijk(:, ids) = OnePlusLogPijRep;
        PijUj = PijUj + Us(:,ids)*(Pijk(:,ids).*OnePlusLogPijRep)';
    end
    P1pluslogP = Pij.*OnePlusLogPij;
    PU = (Us*Pijk').*repmat(sum(P1pluslogP,2)',d,1);
    gradUt = PU - PijUj;
    % Grad for Us
    PijkOnePlusLogPijk = Pijk.*OnePlusLogPijk; % nt x ns;
    PijkUt = Ut*PijkOnePlusLogPijk; % d x ns
    PijkPij = bsxfun(@times, Pijk, sum(P1pluslogP, 2));
    PijkPijUt = Ut*PijkPij; % d x ns
    gradUs = PijkPijUt - PijkUt;
    % Total Grad
    if gpuMode
        Y = gpuArray(single(zeros(d, (ns+nt))));
    else
        Y = single(zeros(d, (ns+nt)));
    end
    Y(:,srcIds) = gradUs;
    Y(:,tgtIds) = gradUt;
    if any(isnan(gather(Y(:))))
        error('gardJE is nan');
    end
end
end
