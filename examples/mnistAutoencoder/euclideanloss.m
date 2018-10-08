function Y = euclideanloss(X, c, dzdy)
%EUCLIDEANLOSS Summary of this function goes here
%   Detailed explanation goes here

assert(numel(X) == numel(c));

d = size(X);

assert(all(d == size(c)));

if nargin == 2 || (nargin == 3 && isempty(dzdy))
    
    Y = 1 / 2 * sum(subsref((X - c) .^ 2, substruct('()', {':'}))); % Y is divided by d(4) in cnn_train.m / cnn_train_mgpu.m.
%     Y = 1 / (2 * prod(d(1 : 3))) * sum(subsref((X - c) .^ 2, substruct('()', {':'}))); % Should Y be divided by prod(d(1 : 3))? It depends on the learning rate.
    
elseif nargin == 3 && ~isempty(dzdy)
    
    assert(numel(dzdy) == 1);
    
    Y = dzdy * (X - c); % Y is divided by d(4) in cnn_train.m / cnn_train_mgpu.m.
%     Y = dzdy / prod(d(1 : 3)) * (X - c); % Should Y be divided by prod(d(1 : 3))? It depends on the learning rate.
    
end

end

