classdef nnmnist < nntest
  properties (TestParameter)
    networkType = {'dagnn', 'simplenn'}
  end

  methods (TestClassSetup)
    function init(test)
      addpath(fullfile(vl_rootnn, 'examples', 'mnist'));
    end
  end

  methods (Test)
    function valErrorRate(test, networkType)
      clear mex ; % will reset GPU, remove MCN to avoid crashing
                  % MATLAB on exit (BLAS issues?)
      if strcmp(test.dataType, 'double'), return ; end
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
      [~, info] = cnn_mnist('train', trainOpts, 'networkType', networkType);
      test.verifyLessThan(info.train.top1err, 0.08);
      test.verifyLessThan(info.val.top1err, 0.025);
    end
  end
end
