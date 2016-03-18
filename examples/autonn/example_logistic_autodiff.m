
run(fullfile(fileparts(mfilename('fullpath')),...
  '..', '..', 'matlab', 'vl_setupnn.m')) ;


% load simple data
s = load('fisheriris.mat') ;
data_x = single(reshape(s.meas.', 1, 1, 4, [])) ;  % features in 3rd channel
[~, ~, data_y] = unique(s.species) ;  % convert strings to class labels



% define inputs
x = Input() ;
y = Input() ;

% predict using a conv layer. create and initialize params automatically
prediction = vl_nnconv(x, 'size', [1, 1, 4, 3]) ;

% define loss, and classification error
loss = vl_nnsoftmaxloss(prediction, y) ;
error = vl_nnloss(prediction, y, 'loss','classerror') ;

% assign names based on workspace variables, and compile net
Layer.autoNames() ;
net = Net(loss + error) ;



% simple SGD
lr = 1e-3 ;
outputs = zeros(1, 100) ;
rng(0) ;
params = [net.params.idx] ;

for iter = 1:100,
  % draw minibatch
  idx = randperm(numel(data_y), 50) ;
  
  net.setInputs('x', data_x(:,:,:,idx), 'y', data_y(idx)) ;
  
  % evaluate network
  net.eval(1) ;
  
  % update weights
  w = net.getValue(params) ;
  dw = net.getDer(params) ;
  
  for k = 1:numel(params),
    w{k} = w{k} - lr * dw{k} ;
  end
  
  net.setValue(params, w) ;
  
  % plot error
  outputs(iter) = net.getValue(error) / numel(idx) ;
end

figure(3) ;
plot(outputs) ;
xlabel('Iteration') ; ylabel('Error') ;

loss
