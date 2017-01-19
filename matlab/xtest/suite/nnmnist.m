classdef nnmnist < nntest
  properties (TestParameter)
    networkType = {'autonn', 'dagnn', 'simplenn'}
  end

  methods (TestClassSetup)
    function init(test)
      addpath(fullfile(vl_rootnn, 'examples', 'mnist'));
      addpath(fullfile(vl_rootnn, 'examples', 'autonn'));
    end
  end

  methods (Test)
    function valErrorRate(test, networkType)
      clear mex ; % will reset GPU, remove MCN to avoid crashing
                  % MATLAB on exit (BLAS issues?)
      if strcmp(test.currentDataType, 'double'), return ; end
      switch test.currentDevice
        case 'cpu'
          gpus = [];
        case 'gpu'
          gpus = 1;
      end
      trainOpts = struct('numEpochs', 1, 'continue', false, 'gpus', gpus, ...
        'plotStatistics', false);
      if strcmp(networkType, 'simplenn')
        trainOpts.errorLabels = {'error', 'top5err'} ;
      end
      
      if ~strcmp(networkType, 'autonn')
        [~, info] = cnn_mnist('train', trainOpts, 'networkType', networkType) ;
        test.verifyLessThan(info.train.error, 0.08);
        test.verifyLessThan(info.val.error, 0.025);
      else
        trainOpts.plotDiagnostics = false ;
        [~, info] = cnn_mnist_autonn('train', trainOpts) ;
        % initialized using xavier's method, not 0.01*randn
        test.verifyLessThan(info.train.error, 0.13);
        test.verifyLessThan(info.val.error, 0.11);
      end
    end
  end
end
