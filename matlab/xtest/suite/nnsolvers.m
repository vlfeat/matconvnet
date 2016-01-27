classdef nnsolvers < nntest
  properties (TestParameter)
%     networkType = {'simplenn', 'dagnn'}
    networkType = {'simplenn'}
    solver = {'sgd', 'adagrad', 'adadelta'}
  end
  properties
    imdb
    init_w
    init_b
  end

  methods (TestClassSetup)
    function data(test)
      % synthetic data, 2 classes of gaussian samples with different means
      sz = [15, 10, 5] ;
      mean1 = test.randn(sz) ;
      mean1 = 2 * mean1 / norm(mean1(:)) ;
      mean2 = test.randn(sz) ;
      mean2 = 3 * mean2 / norm(mean2(:)) ;
      
      x1 = bsxfun(@plus, mean1, test.randn([sz, 100])) ;
      x2 = bsxfun(@plus, mean2, test.randn([sz, 100])) ;
      test.imdb.x = cat(4, x1, x2) ;
      test.imdb.y = [test.ones(100, 1); -test.ones(100, 1)] ;
      
      test.init_w = test.randn([sz, 2]) ;  % initial parameters
      test.init_b = test.zeros([2, 1]) ;
    end
  end

  methods (Test)
    function basic(test, networkType, solver)
      clear mex ; % will reset GPU, remove MCN to avoid crashing
                  % MATLAB on exit (BLAS issues?)
      if strcmp(test.dataType, 'double'), return ; end

      % a simple logistic regression network
      net.layers = {struct('type','conv', 'weights',{{test.init_w, test.init_b}}), ...
                    struct('type','softmaxloss')} ;
      
      switch networkType
        case 'simplenn',
          trainfn = @cnn_train ;
        case 'dagnn',
          trainfn = @cnn_train_dag ;
          net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;
          net.addLayer('error', dagnn.Loss('loss', 'classerror'), ...
                      {'prediction','label'}, 'error') ;
      end

      switch test.currentDevice
        case 'cpu', gpus = [];
        case 'gpu', gpus = 1;
      end

      % train 1 epoch with small batches and check convergence
      [~, info] = trainfn(net, test.imdb, ...
        @(imdb, batch) deal(imdb.x(:,:,:,batch), imdb.y(batch)), ...
        'train', 1:numel(test.imdb.y), 'val', 1, ...
        'solver',solver, 'batchSize', 10, 'numEpochs',1, 'continue', false, ...
        'gpus', gpus, 'plotStatistics', false) ;

      test.verifyLessThan(info.train.error(1), 0.3);
      test.verifyLessThan(info.train.objective, 1.4e+5);
    end
  end
end
