function [iu, p_ij] = vl_jaccardindex(s1, s2)
% Compute the Jaccard index between two sets.
% s1 and s2 are column vectors. Element 0's are ignored.

if size(s1,1) == 1, s1 = s1(:); end
if size(s2,1) == 1, s2 = s2(:); end

max1 = max(s1);
max2 = max(s2);
p_ij = accumarray(horzcat(s1, s2), 1, [max1 max2]);
p_i = sum(p_ij, 2);
p_j = sum(p_ij, 1);
iu = p_ij ./ (p_i + p_j - p_ij);
p_ij = p_ij / sum(p_i);
