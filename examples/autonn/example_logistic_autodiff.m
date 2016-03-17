
run(fullfile(fileparts(mfilename('fullpath')),...
  '..', '..', 'matlab', 'vl_setupnn.m')) ;


% load simple data
s = load('fisheriris.mat') ;
data_x = single(s.meas.') ;  % features-by-samples matrix
[~, ~, data_y] = unique(s.species) ;  % convert strings to class labels



% define inputs and params
x = Input() ;
y = Input() ;
w = Param('value', 0.01 * randn(3, 4, 'single')) ;
b = Param('value', 0.01 * randn(3, 1, 'single')) ;

% combine them using math operators, which define the prediction
prediction = w * x + b ;

% reshape into a 4D tensor (which softmaxloss expects) and compute loss
loss = vl_nnsoftmaxloss(reshape(prediction, {1, 1, 3, []}), y) ;

% assign names based on workspace variables, and compile net
Layer.autoNames() ;
net = Net(loss) ;



% simple SGD
lr = 1e-3 ;
outputs = zeros(1, 100) ;
rng(0) ;

for iter = 1:100,
  % draw minibatch
  idx = randperm(numel(data_y), 50) ;
  
  net.setValue(x, data_x(:,idx)) ;
  net.setValue(y, data_y(idx)) ;
  
  % evaluate network
  net.eval(1) ;
  
  % update weights
  net.setValue(w, net.getValue(w) - lr * net.getDer(w)) ;
  net.setValue(b, net.getValue(b) - lr * net.getDer(b)) ;
  
  % plot loss
  outputs(iter) = net.getValue(loss) ;
end

figure(3) ;
plot(outputs) ;
xlabel('Iteration') ; ylabel('Loss') ;

loss
