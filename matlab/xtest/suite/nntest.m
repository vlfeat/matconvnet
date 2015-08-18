classdef nntest < matlab.unittest.TestCase
  properties (MethodSetupParameter)
    device = {'cpu', 'gpu'}
  end

  properties
    randn
    rand
    toDevice
    range = 128
  end

  methods (TestMethodSetup)
    function generators(test, device)
      range = 128 ;
      seed = 0 ;
      switch device
        case 'gpu'
          gpuDevice ;
          test.randn = @(varargin) range * gpuArray.randn(varargin{:}) ;
          test.rand = @(varargin) range * gpuArray.rand(varargin{:}) ;
          test.toDevice = @(x) gpuArray(x) ;
          parallel.gpu.rng(seed, 'combRecursive') ;
        case 'cpu'
          test.randn = @(varargin) range * randn(varargin{:}) ;
          test.rand = @(varargin) range * rand(varargin{:}) ;
          test.toDevice = @(x) gather(x) ;
          rng(seed, 'combRecursive') ;
      end
    end
  end

  methods
    function der(test, g, x, dzdy, dzdx, delta, tau)
      if nargin < 7
        tau = [] ;
      end
      dzdx_ = test.numder(g, x, dzdy, delta) ;
      test.eq(gather(dzdx_), gather(dzdx), tau) ;
    end

    function eq(test,a,b,tau)
      a = gather(a) ;
      b = gather(b) ;
      if nargin > 3 && ~isempty(tau) && tau < 0
        tau_min = -tau ;
        tau = [] ;
      else
        tau_min = 0 ;
      end
      if nargin < 4 || isempty(tau)
        maxv = max([max(a(:)), max(b(:))]) ;
        minv = min([min(a(:)), min(b(:))]) ;
        tau = max(1e-2 * (maxv - minv), 1e-3 * max(maxv, -minv)) ;
      end
      tau = max(tau, tau_min) ;
      if isempty(tau) % can happen if a and b are empty
        tau = 0 ;
      end
      tol = matlab.unittest.constraints.AbsoluteTolerance(single(tau)) ;
      test.verifyThat(a, ...
        matlab.unittest.constraints.IsEqualTo(b, 'Within', tol)) ;
    end
  end

  methods (Static)
    function dzdx = numder(g, x, dzdy, delta)
      if nargin < 3
        delta = 1e-3 ;
      end
      dzdy = gather(dzdy) ;
      y = gather(g(x)) ;
      dzdx = zeros(size(x),'double') ;
      for i=1:numel(x)
        x_ = x ;
        x_(i) = x_(i) + delta ;
        y_ = gather(g(x_)) ;
        factors = dzdy .* (y_ - y)/delta ;
        dzdx(i) = dzdx(i) + sum(factors(:)) ;
      end
      dzdx = single(dzdx) ;
    end
  end
end
