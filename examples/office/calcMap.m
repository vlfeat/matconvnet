function [map, succRate] = calcMap(Us, Ut, srcLabels, tgtLabels)
% This funtion calculates the mean average precision for image recall.
% Input:
%       Us = Source hash code (real values) [d x ns]
%       Ut = Target hash code (real values) [d x nt]
%       srcLabels = source labels [1 x ns]
%       tgtLabels = target labels [1 x nt]

% Copyright (C) 2016 Hemanth Venkateswara.
% All rights reserved.

S = getSimilarityMatrix(srcLabels, tgtLabels);
Bs = Us > 0;
Bt = Ut > 0;
[~, orderH] = calcHammingRank(Bs, Bt);
[map, succRate] = getMap(orderH, S');

end

function [S] = getSimilarityMatrix(srcLabels, tgtLabels)
% Calculates the [ns x nt] similarity matrix where Sij = 1 iff i and j
% belong to the same category

Dist = repmat(srcLabels', 1, length(tgtLabels)) - repmat(tgtLabels, length(srcLabels), 1);
S = Dist == 0;
end

function [distH, orderH] = calcHammingRank(Bs, Bt)
% Calculate the Hamming distance between each target hash and the source
% hash and order them in increasing order of distance.
distH = calcHammingDist(Bt, Bs);
[~, orderH] = sort(distH, 2, 'ascend');
end

function [distH] = calcHammingDist(Bt, Bs)
% Calculate the Hamming distance between each target hash and the source
% hash

% convert to [-1, +1] hash
Pt = sign(Bt - 0.5);
Ps = sign(Bs - 0.5);
d = size(Bs, 1);
distH = round((d - Pt'*Ps) / 2);
end

function [map, succRate] = getMap(orderH, S)
% Function to calculate the map
[nt, ns] = size(S);
pos = 1:ns;
map = 0;
numSucc = 0;
for i = 1:nt
    nbr = S(i, orderH(i, :));
    nRel = sum(nbr);
    if nRel > 0
        prec = cumsum(nbr) ./ pos;
        ap = mean(prec(nbr));
        map = map + ap;
        numSucc = numSucc + 1;
    end
end
map = map/numSucc;
succRate = numSucc/nt;
end