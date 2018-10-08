function Y = sigmoidcrossentropyloss(X, c, dzdy)
%EUCLIDEANLOSS Summary of this function goes here
%   Detailed explanation goes here

assert(numel(X) == numel(c));

d = size(X);

assert(all(d == size(c)));

p     = sigmoid(c);
p_hat = sigmoid(X);

if nargin == 2 || isempty(dzdy)
    
    Y = -sum(subsref(p * log(p_hat) + (1 - p) * log(1 - p_hat), substruct('()', {':'}))); % Y is divided by d(4) in cnn_train.m / cnn_train_mgpu.m.
%     Y = -1 / prod(d(1 : 3)) * sum(subsref(p * log(p_hat) + (1 - p) * log(1 - p_hat), substruct('()', {':'}))); % Should Y be divided by prod(d(1 : 3))? It depends on the learning rate.
    
elseif nargin == 3 && ~isempty(dzdy)
    
    assert(numel(dzdy) == 1);
    
    Y = dzdy * (p_hat - p); % Y is divided by d(4) in cnn_train.m / cnn_train_mgpu.m.
%     Y = dzdy / prod(d(1 : 3)) * (p_hat - p); % Should Y be divided by prod(d(1 : 3))? It depends on the learning rate.
    
end

end

