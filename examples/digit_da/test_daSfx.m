function[] = test_daSfx()
% TEST_DASFX Evaluate Classification error with Domain Adaptive
% softmaxloss_entropyloss

% Copyright (C) 2016 Hemanth Venkateswara.
% All rights reserved.

src = 'mnist';
tgt = 'usps';
srcNetFile = 'net-epoch-40';

opts.gpus = 1;
opts.numThreads = 12;
opts.boostSize = 200;
opts.saveData = true; % flag to save the feature representation.

ext = '.mat';
imdbFile = 'imdb.mat';
src_tgt = [src, '_', tgt];

fprintf('Results for %s to %s using file %s\n', src, tgt, srcNetFile);

dataRoot = '/data/DB/digit_da/';
imDataRoot = '/home/ASUAD/hkdv1/CodeRep/MatConvNet/da_hash/examples/digit_da';

netPath = [dataRoot, src_tgt, '-vgg-daSfx/', srcNetFile, ext];
imdbPath = [dataRoot, src_tgt, '-vgg-daSfx/', imdbFile];
opts.saveFilePath = [dataRoot, src_tgt, '-vgg-daSfx/','fortSNE.mat'];
net = load(netPath);
net = net.net;

opts.srcDataDir = fullfile(imDataRoot, src) ;
opts.tgtDataDir = fullfile(imDataRoot, tgt) ;
opts.srcMetaPath = fullfile(imDataRoot, 'meta.mat');
opts.valSizeRatio = 0;
% imdb = createEntropyImdb(opts) ;
imdb = load(imdbPath);

srcIds = imdb.images.label > 0;
tgtIds = imdb.images.label == 0;
tgtLabels = double(imdb.images.actLabel(tgtIds));
tgtFileNames = imdb.images.trainValNames(tgtIds);

trIds = imdb.images.set==1;
valIds = imdb.images.set==2;

% Use Train from source
trSrcLabels = imdb.images.actLabel(srcIds & trIds);
trSrcFileNames = imdb.images.trainValNames(srcIds & trIds);
% % Use Val from source
% trSrcLabels = imdb.images.actLabel(srcIds & valIds);
% trSrcFileNames = imdb.images.trainValNames(srcIds & valIds);

clNames = imdb.classes.description;
srcFileNames = {};
srcLabels = {};
srcKSet = {};
for ii = 1:length(clNames)
    ids = find(trSrcLabels == ii);
    ids = ids(1:min(opts.boostSize, length(ids)));
    srcKSet{end+1} = length(ids);
    srcFileNames{end+1} = trSrcFileNames(ids);
    srcLabels{end+1} = trSrcLabels(ids);
end
srcFileNames = cat(2, srcFileNames{:});
srcLabels = double(cat(2, srcLabels{:}));
imFileNames = {srcFileNames{:}, tgtFileNames{:}};
imLabels =cat(2, srcLabels, tgtLabels);
srcTgtSet = [ones(1, length(srcLabels)), 2*ones(1, length(tgtLabels))];
batchSize = 1000;
U = [];
for t = 1:batchSize:length(imLabels)
    batch = t:min(t+batchSize-1, numel(imLabels));
    imFilesNamesBatch = imFileNames(batch);
    imLabelsBatch = imLabels(batch);
    [hashOut] = getHash(net, imFilesNamesBatch, imLabelsBatch, net.meta, opts);
    U = [U, hashOut];
end

fprintf('%s -> %s epochs = %s\n', src, tgt, srcNetFile);

opts.numClasses = length(net.meta.classes.name);
[predLab] = boostedPrediction(U, imLabels, srcTgtSet, srcKSet, opts);

acc = sum(predLab==tgtLabels)/length(tgtLabels)*100;
fprintf('\n%s -> %s: Acc = %0.3f\n', src, tgt, acc);

end

% -------------------------------------------------------------------------
function[U] = getHash(net_cpu, imFileNames, imLabels, meta, opts)
% -------------------------------------------------------------------------
% This function appends the srcFilesNames and tgtIm and pre-processes it 
% for input to the CNN

numGpus = opts.gpus;
% bs = net_cpu.meta.trainOpts.batchSize;
bs = 500;
evalMode = 'test';

U = [];
if numGpus >= 1
    ims = getBatch(imFileNames, meta, opts) ;
    net = vl_simplenn_move(net_cpu, 'gpu');
    for t = 1:bs:length(imLabels)
        batch = t:min(t+bs-1, numel(imLabels));
        net.layers{end}.class = imLabels(batch);
        imbatch = ims(:,:,:,batch);
        res = vl_simplenn_dah(net, imbatch, [], [], 'mode', evalMode);
        U = [U, squeeze(gather(res(end-1).x))];
        fprintf('\n');
    end
else
    ims = gather(getBatch(imFileNames, meta, opts)) ;
    for t = 1:bs:length(imLabels)
        batch = t:min(t+bs-1, numel(imLabels));
        net_cpu.layers{end}.class = imLabels(batch);
        imbatch = ims(:,:,:,batch);
        res = vl_simplenn_dah(net_cpu, imbatch, [], [], 'mode', evalMode);
        U = [U, squeeze(gather(res(end-1).x))];
        fprintf('\n');
    end
end
U = double(U);
end

% -------------------------------------------------------------------------
function[ims] = getBatch(imFileNames, meta, opts)
% -------------------------------------------------------------------------
if numel(meta.normalization.averageImage) == 1
  mu = double(meta.normalization.averageImage) ;
elseif numel(meta.normalization.averageImage) == 3
  mu = double(meta.normalization.averageImage(:)) ;
else
  mu = imresize(single(meta.normalization.averageImage), ...
                meta.normalization.imageSize(1:2)) ;
end

useGpu = numel(opts.gpus) > 0 ;

bopts.test = struct(...
  'useGpu', useGpu, ...
  'numThreads', opts.numThreads, ...
  'imageSize',  meta.normalization.imageSize(1:2), ...
  'cropSize', meta.normalization.cropSize, ...
  'subtractAverage', mu) ;
ims = getImageBatch(imFileNames, bopts.test) ;
end

% -------------------------------------------------------------------------
function[predLab] = boostedPrediction(U, imLabels, srcTgtSet, srcKSet, opts)
% -------------------------------------------------------------------------
% This function estimates the label of the target using boosted CNN
%  
%  **************         **************
%  **************  NOTE   **************
%  **************         **************
%  We try different ways to estimate the target labels and we list the
%  accuracies using all of the different techniques. 
%  'Boosted Target Acc2' is our choice

srcIds = srcTgtSet==1;
tgtIds = srcTgtSet==2;
Us = U(:, srcIds);
Ut = U(:, tgtIds);
srcLabels = imLabels(srcIds);
tgtLabels = imLabels(tgtIds);

if opts.saveData
    save(opts.saveFilePath, 'U', 'imLabels', 'srcTgtSet');
end
numClasses = opts.numClasses;

% Source Prediction
Usdot = Us'*Us;
expUsdot = exp(-0.5*Usdot);
expUsdot(isinf(expUsdot)) = 1e30;
% As = 1./(1 + exp(-0.5*Usdot));
As = 1./(1 + expUsdot);

Ys = zeros(numClasses, size(Usdot,2));
count = 0;
for kk = 1:length(srcKSet)
    idk = count+1 : count+srcKSet{kk};
    Ys(kk, :) = sum(As(idk, :));
    count = count + srcKSet{kk};
end
% As = reshape(As, opts.boostSize, []);
% Y = sum(As, 1);
% Y = reshape(Y, numClasses, []);
[bestScore, predS] = max(Ys);
srcAcc = sum(predS==srcLabels)/length(srcLabels)*100;
fprintf('\nSource Training Accuracy = %.03f\n', srcAcc);

Ustdot = Us'*Ut;
expUstdot = exp(-0.5*Ustdot);
expUstdot(isinf(expUstdot)) = 1e30;
% Ast = 1./(1 + exp(-0.5*Ustdot)); % 0.5*<ui uj> % [ns x nt]
Ast = 1./(1 + expUstdot); % 0.5*<ui uj> % [ns x nt]

if any(isnan(Ast(:)))
    error('Ast is nan');
end
% if boostSize = 5 and numClasses = 31, ns = 31*5 = 155
% Ast = [ns x nt]
Yt = zeros(numClasses, size(Ustdot,2));
count = 0;
for kk = 1:length(srcKSet)
    idk = count+1 : count+srcKSet{kk};
    Yt(kk, :) = sum(Ast(idk, :));
    count = count + srcKSet{kk};
end
[bestScore, pred1] = max(Yt);
acc = sum(pred1==tgtLabels)/length(tgtLabels)*100;
fprintf('Simple Acc1 = %0.3f\n', acc);

Usc = zeros(size(U,1), numClasses); % d x C
K = opts.boostSize;
count = 0;
for ii = 1:numClasses
    K = srcKSet{ii};
    onesk = ones(K,1)/K;
    ids = count+1 : count+K;
    Usc(:,ii) = U(:, ids)*onesk;
    count = count + K;
end

UsUscdot = Us'*Usc; % nt x C
UsUscdot = bsxfun(@minus, UsUscdot, max(UsUscdot,[],2));
P = exp(UsUscdot);
P = bsxfun(@rdivide, P, sum(P,2)); % nt x C
[bestScore, predLab] = max(P');
boostedSAcc = sum(predLab==srcLabels)/length(srcLabels)*100;
fprintf('Boosted Source Acc2 = %0.3f\n', boostedSAcc);

UtUsdot = Ut'*Us; % nt x C
UtUsdot = bsxfun(@minus, UtUsdot, max(UtUsdot,[],2));
P = exp(UtUsdot);
Pij = zeros(size(Ut,2), numClasses);
count = 0;
for ii = 1:numClasses
    K = srcKSet{ii};
    onesk = ones(K,1);
    ids = count+1 : count+K;
    Pij(:,ii) = P(:, ids)*onesk;
    count = count + K;
end
P = bsxfun(@rdivide, Pij, sum(Pij,2)); % nt x C
[bestScore, predLab] = max(P');
boostedTAcc = sum(predLab==tgtLabels)/length(tgtLabels)*100;
fprintf('Boosted Target Acc2 = %0.3f\n', boostedTAcc);

% Boosted target with hash values
Bt = sign(Ut);
% Bt(Bt==0) = 1;
Bs = sign(Us);
% Bs(Bs==0) = 1;
BtBsdot = Bt'*Bs; % nt x C
BtBsdot = bsxfun(@minus, BtBsdot, max(BtBsdot,[],2));
Pb = exp(BtBsdot);
Pbij = zeros(size(Bt,2), numClasses);
count = 0;
for ii = 1:numClasses
    K = srcKSet{ii};
    onesk = ones(K,1);
    ids = count+1 : count+K;
    Pbij(:,ii) = Pb(:, ids)*onesk;
    count = count + K;
end
Pb = bsxfun(@rdivide, Pbij, sum(Pbij,2)); % nt x C
[bestScore, predLabB] = max(Pb');
boostedHTAcc = sum(predLabB==tgtLabels)/length(tgtLabels)*100;
fprintf('Hash Boosted Target Acc2 = %0.3f\n', boostedHTAcc);

Ustdot = Us'*Ut;
Ast = exp(0.5*Ustdot);
Ast(isinf(Ast)) = 1e30;
Yt = zeros(numClasses, size(Ustdot,2));
count = 0;
for kk = 1:length(srcKSet)
    idk = count+1 : count+srcKSet{kk};
    Yt(kk, :) = sum(Ast(idk, :));
    count = count + srcKSet{kk};
end
[bestScore, pred3] = max(Yt);
acc = sum(pred3==tgtLabels)/length(tgtLabels)*100;
fprintf('Complex Boosted Acc3 = %0.3f\n', acc);

Ustdot = Us'*Ut;
UsNorm = repmat(sqrt(sum(Us.^2))', 1, size(Ut,2));
UtNorm = repmat(sqrt(sum(Ut.^2)), size(Us,2), 1);
Ast = (Ustdot./UsNorm)./UtNorm;
Yt = zeros(numClasses, size(Ustdot,2));
count = 0;
for kk = 1:length(srcKSet)
    idk = count+1 : count+srcKSet{kk};
    Yt(kk, :) = sum(Ast(idk, :));
    count = count + srcKSet{kk};
end
[bestScore, pred3] = max(Yt);
acc = sum(pred3==tgtLabels)/length(tgtLabels)*100;
fprintf('Cosine Boosted Acc3 = %0.3f\n', acc);

Ast = sign(Us)'*sign(Ut);
Yt = zeros(numClasses, size(Ut,2));
count = 0;
for kk = 1:length(srcKSet)
    idk = count+1 : count+srcKSet{kk};
    Yt(kk, :) = sum(Ast(idk, :));
    count = count + srcKSet{kk};
end
[bestScore, pred3] = max(Yt);
acc = sum(pred3==tgtLabels)/length(tgtLabels)*100;
fprintf('Sign Boosted Acc3 = %0.3f\n', acc);

Mdl = fitcknn(Us', srcLabels, 'NumNeighbors', 1);
pred = predict(Mdl, Ut');
oneNNacc = sum(pred'==tgtLabels)/length(tgtLabels)*100;
fprintf('1-NN Acc = %0.3f\n', oneNNacc);

Mdl = fitcknn(Us', srcLabels, 'NumNeighbors', 1, 'Distance', @hammingDist);
pred = predict(Mdl, Ut');
hamm1acc = sum(pred'==tgtLabels)/length(tgtLabels)*100;
fprintf('Hamming 1-kNN Acc = %0.3f\n', hamm1acc);

Mdl = fitcknn(Us', srcLabels, 'NumNeighbors', 10, 'Distance', @hammingDist);
pred = predict(Mdl, Ut');
hamm10acc = sum(pred'==tgtLabels)/length(tgtLabels)*100;
fprintf('Hamming 10-kNN Acc = %0.3f\n', hamm10acc);

Ustdot = Us'*Ut;
[bestScore, predId] = max(Ustdot);
pred = srcLabels(predId);
acc = sum(pred==tgtLabels)/length(tgtLabels)*100;
fprintf('One Sample Acc = %0.3f\n', acc);

[srcMap, succRate] = calcMap(Us, Us, srcLabels, srcLabels);
fprintf('Source MAP = %0.3f, Succ Rate = %0.3f\n', srcMap, succRate);

[tgtMap, succRate] = calcMap(Us, Ut, srcLabels, tgtLabels);
fprintf('Target MAP = %0.3f, Succ Rate = %0.3f\n', tgtMap, succRate);

[~, predS] = max(Us);
accsSfx = sum(predS==srcLabels)/length(srcLabels)*100;
fprintf('Softmax Source Acc = %0.3f\n', accsSfx);

[~, predT] = max(Ut);
accttSfx = sum(predT==tgtLabels)/length(tgtLabels)*100;
fprintf('Softmax Target Acc = %0.3f\n', accttSfx);

fprintf('BoostedSrcAcc %0.2f, BoostedTgtAcc %0.2f, 1-NN %0.2f, Hamm1NN %0.2f, Hamm10NN %0.2f, srcMap %0.3f, tgtMap %0.3f Sfx_SrcAcc %0.3f, Sfx_TgtAcc %0.3f\n', ...
    boostedSAcc, boostedTAcc, oneNNacc, hamm1acc, hamm10acc, srcMap, tgtMap, accsSfx, accttSfx);
fprintf('BoostedSrcAcc, BoostedTgtAcc, 1-NN, Hamm1NN, Hamm10NN, srcMap, tgtMap, Sfx_SrcAcc, Sfx_TgtAcc\n');
fprintf('%0.2f, %0.2f, %0.2f, %0.2f, %0.2f, %0.3f, %0.3f, %0.3f, %0.3f\n', ...
    boostedSAcc, boostedTAcc, oneNNacc, hamm1acc, hamm10acc, srcMap, tgtMap, accsSfx, accttSfx);

end

function [hammD] = hammingDist(x, Z)
% Calculate hamming Distance
% x = [1 x d] vector
% Z = [n x d] matrix of n vectors
% hammD = [n x 1] vector of hamming distances
x = x>0;
Z = Z>0;
Px = sign(x - 0.5);
Pz = sign(Z - 0.5);
d = length(x);
hammD = round((d - Px*Pz') / 2);
hammD = hammD';
end
