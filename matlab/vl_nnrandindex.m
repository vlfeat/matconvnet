function [thisIndex, thisInfo] = vl_nnrandindex(S, T, varargin)
% Returns an array of size batchSize x 1

opts.thin = 'watershed';
opts.conn = 4;
opts.computeRandIndex = true;
opts.computeRandInfo = true;
[opts, ~] = vl_argparse(opts, varargin);

thisIndex = zeros(size(S,4), 1);
thisInfo = zeros(size(S,4), 1);
for i = 1:size(S,4)
    [index, info] = compute_rand_f_score(S(:,:,:,i), T(:,:,:,i), opts);
    thisIndex(i) = index;
    thisInfo(i) = info;
end

% ------------------------------------------------------------------------------
function [index, info] = compute_rand_f_score(S, T, opts)
% compute fore-ground restricted rand f-score
[index, info] = deal(nan);
if ~opts.computeRandIndex && ~opts.computeRandInfo, return; end

if strcmpi(opts.thin, 'morph')
    S = bwmorph(S, 'thicken', Inf);
elseif strcmpi(opts.thin, 'watershed')
    S = watershed_thinning(S, opts.conn);
end

% S and T are logical
LS = bwlabel(S, opts.conn);
LT = bwlabel(T, opts.conn);
nS = max(LS(:));
nT = max(LT(:));
p = accumarray([LT(:), LS(:)]+1, 1, [nT nS]+1);
p_ = p(2:end,:);
n = sum(sum(p_));
p_ = p_/n;
p__ = p_(:,2:end);
pi0 = p_(:,1);
aux = sum(pi0);
ai = [0; sum(p_, 2)];
bj = [0, sum(p__, 1)];

if opts.computeRandIndex
    sumA2 = sum(ai.^2);
    sumB2 = sum(bj.^2) + aux/n;
    sumAB2 = sum(sum(p__.^2)) + aux/n;
    prec = sumAB2/sumB2;
    rec = sumAB2/sumA2;
    index = 2/(1/prec+1/rec);
end

if opts.computeRandInfo
    ok = ai~=0; sumA = sum(ai(ok).*log(ai(ok)));
    ok = bj~=0; sumB = sum(bj(ok).*log(bj(ok))) - aux*log(n);
    ok = p__~=0; sumAB = sum(sum(p__(ok).*log(p__(ok)))) - aux*log(n);
    % H(A)
    ha = -sumA;
    % H(B)
    hb = -sumB;
    % H(A|B)
    hab = sumB - sumAB;
    % H(B|A)
    hba = sumA - sumAB;
    % An information theoretic analog of Rand precision
    % is the asymmetrically normalized mutual information
    % C(A|B) (A = ground truth, B = segmentation)
    prec = (ha - hab) / ha;
    % An information theoretic analog of Rand recall
    % is defined similarly
    % C(B|A) (A = ground truth, B = segmentation)
    rec = (hb - hba) / hb;
    if ha == 0
        prec = 0.0;
        rec = 1.0;
    end
    if hb == 0
        prec = 1.0;
        rec = 0.0;
    end
    % F-score
    info = 2.0 * prec * rec / (prec + rec);
end

% ------------------------------------------------------------------------------
function bw = watershed_thinning(bw, conn)
bd = ~bw;
ws = watershed(bd, conn);
bd = ws == 0;
bw = ~bd;
