classdef nnpdist < nntest
  properties (TestParameter)
    oneToOne = {false, true}
    noRoot = {false, true}
    p = {.5 1 2 3}
  end
  methods (Test)
    function basic(test,oneToOne, noRoot, p)
      h = 13 ;
      w = 17 ;
      d = 4 ;
      n = 5 ;
      x = test.randn(h,w,d,n,'single') ;
      if oneToOne
        x0 = test.randn(h,w,d,n,'single') ;
      else
        x0 = test.randn(1,1,d,n) ;
      end
      y = vl_nnpdist(x, x0, p, 'noRoot',noRoot) ;

      % make sure they are not too close in anyd dimension as
      % this may be a problem for the finite difference
      % dereivatives as one could approach0 which is not
      % differentiable for some p-norms

      s = abs(bsxfun(@minus, x, x0)) < 5*test.range*1e-3 ;
      x(s) = x(s) + 5*test.range ;

      dzdy = test.rand(h, w, 1, n) ;
      dzdx = vl_nnpdist(x,x0,p,dzdy,'noRoot',noRoot) ;
      test.der(@(x) vl_nnpdist(x,x0,p,'noRoot',noRoot), x, dzdy, dzdx, test.range * 1e-4) ;
    end
  end
end
