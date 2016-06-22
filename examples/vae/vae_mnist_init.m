function net = vae_mnist_init(varargin)
% VAE_MNIST_INIT Initialize a VAE for MNIST
opts.trainMethod='rmsprop';
opts.eps_std=0.01;
opts = vl_argparse(opts, varargin) ;

rng('default');
rng(0) ;
net = dagnn.DagNN() ;
net.conserveMemory=0;
convBlock1 = dagnn.Conv('size', [1 1 784 128], 'hasBias', true) ;
convBlock2 = dagnn.Conv('size', [1 1 128 2], 'hasBias', true) ;
convBlock3 = dagnn.Conv('size', [1 1 2 128], 'hasBias', true) ;
convBlock4 = dagnn.Conv('size', [1 1 128 784], 'hasBias', true) ;
reluBlock = dagnn.ReLU() ;

net.addLayer('x2h', convBlock1, {'input'}, {'h1'}, {'c1f', 'c1b'}) ;
net.addLayer('relu1', reluBlock, {'h1'}, {'h'}, {}) ;
net.addLayer('h2z_mean', convBlock2, {'h'}, {'z_mean'}, {'c2f', 'c2b'}) ;
net.addLayer('h2z_log_std', convBlock2, {'h'}, {'z_log_std'}, {'c3f', 'c3b'}) ;
net.addLayer('z2z', dagnn.Gaussian(opts.eps_std), {'z_mean','z_log_std'}, {'z'}, {}) ;
net.addLayer('z2hhat', convBlock3, {'z'}, {'hhat1'}, {'c4f', 'c4b'}) ;
net.addLayer('relu2', reluBlock, {'hhat1'}, {'hhat'}, {}) ;
net.addLayer('hhat2xhat', convBlock4, {'hhat'}, {'xhat1'}, {'c5f', 'c5b'}) ;
net.addLayer('sigmoid1', dagnn.Sigmoid(), {'xhat1'}, {'xhat'}, {}) ;
net.addLayer('loss', dagnn.VaeLoss(),{'xhat','input','z_mean','z_log_std'}, {'objective'}, {}) ;
net.initParams() ;

if strcmpi(opts.trainMethod,'rmsprop')
    for p=1:numel(net.params)
        net.params(p).trainMethod='rmsprop';
    end
end

% Meta parameters
net.meta.inputSize = [1 1 784] ;
net.meta.trainOpts.learningRate = 0.001 ;
net.meta.trainOpts.weightDecay= 0.99;
net.meta.trainOpts.numEpochs = 20 ;
net.meta.trainOpts.batchSize = 16 ;