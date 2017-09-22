classdef nnnormalizelp < nntest
  properties (TestParameter)
    h = {1 2 3 4}
    w = {1 2 3 4}
    d = {2 3 4}
    p = {2 4}
  end

  methods (Test)
    function basicl2(test, h,w,d)
      x = test.randn(h,w,d,3) ;
      y = vl_nnnormalizelp(x) ;
      dzdy = test.rand(size(y))-0.5 ;
      dzdx = vl_nnnormalizelp(x,dzdy) ;
      test.der(@(x) vl_nnnormalizelp(x), x, dzdy, dzdx, 1e-4, 0.3) ;
    end

    function lp(test, p)
      x = test.randn(2,3,5,3) / test.range ;
      y = vl_nnnormalizelp(x,'p', p) ;
      dzdy = test.rand(size(y))-0.5 ;
      dzdx = vl_nnnormalizelp(x,dzdy, 'p', p) ;
      test.der(@(x) vl_nnnormalizelp(x,'p',p), x, dzdy, dzdx, 1e-4, 0.3) ;
    end

    function dimensions(test, p)
      x = test.randn(2,3,5,3) / test.range ;
      dimsr =  {[1],[2],[3],[4], ...
                [1 2], [1 3], [1 4], [2 3], [2 4], [3 4], ...
                [1 2 3], [2 3 4], ...
                [1 2 3 4]} ;
      for dims = dimsr
        opts = {'p',2,'dimensions',dims{1},'epsilon',1e-4} ;
        y = vl_nnnormalizelp(x,opts{:}) ;
        dzdy = test.rand(size(y))-0.5 ;
        dzdx = vl_nnnormalizelp(x,dzdy,opts{:}) ;
        test.der(@(x) vl_nnnormalizelp(x,opts{:}), x, dzdy, dzdx, .5e-4, 1.0) ;
      end
    end
  end
end
