function y = sigmoid(x, dzdy)
%SIGMOID Summary of this function goes here
%   Detailed explanation goes here

y = 1 ./ (1 + exp(-x));

if nargin == 2 && ~isempty(dzdy)
    
    assert(all(size(x) == size(dzdy)));
    
    y = dzdy .* y .* (1 - y);
    
end

end

