classdef nnbnorm < nntest
  properties (TestParameter)
    rows = {2 8 13}
    cols = {2 8 17}
    numDims = {1 3 4}
    batchSize = {2 7}
  end
  methods (Test)
    function regression(test, rows,cols, numDims, batchSize)
      r = rows ;
      c = cols ;
      nd = numDims ;
      bs = batchSize ;
      x = test.randn(r, c, nd, bs) ;
      g = test.randn(nd, 1) / test.range ;
      b = test.randn(nd, 1) / test.range ;
      epsilon = 0.001 ;

      [y,m] = vl_nnbnorm(x,g,b,'epsilon',epsilon) ;
      n = numel(x) / size(x,3) ;
      mu = sum(sum(sum(x,1),2),4) / n ;
      sigma2 = sum(sum(sum(bsxfun(@minus,x,mu).^2,1),2),4) / n + epsilon ;
      sigma = sqrt(sigma2) ;
      m_ = [mu(:),sigma(:)] ;
      test.eq(m,m_) ;

      g = reshape(g,1,1,[]) ;
      b = reshape(b,1,1,[]) ;
      y_ = bsxfun(@plus,b,bsxfun(@times,g,bsxfun(@rdivide,bsxfun(@minus,x,mu),sigma))) ;
      test.eq(y,y_) ;
    end

    function basic(test, rows, cols, numDims, batchSize)
      r = rows ;
      c = cols ;
      nd = numDims ;
      bs = batchSize ;
      x = test.randn(r, c, nd, bs) ;
      %g = test.randn(1, 1, nd, 1) ;
      %b = test.randn(1, 1, nd, 1) ;
      g = test.randn(nd, 1) / test.range ;
      b = test.randn(nd, 1) / test.range ;

      y = vl_nnbnorm(x,g,b) ;
      dzdy = test.randn(size(y)) ;
      [dzdx,dzdg,dzdb] = vl_nnbnorm(x,g,b,dzdy) ;

      test.der(@(x) vl_nnbnorm(x,g,b), x, dzdy, dzdx, test.range * 1e-3) ;
      test.der(@(g) vl_nnbnorm(x,g,b), g, dzdy, dzdg, 1e-2) ;
      test.der(@(b) vl_nnbnorm(x,g,b), b, dzdy, dzdb, 1e-3) ;
    end

    function givenMoments(test, rows, cols, numDims, batchSize)
      r = rows ;
      c = cols ;
      nd = numDims ;
      bs = batchSize ;
      x = test.randn(r, c, nd, bs) ;
      %g = test.randn(1, 1, nd, 1) ;
      %b = test.randn(1, 1, nd, 1) ;
      g = test.randn(nd, 1) / test.range ;
      b = test.randn(nd, 1) / test.range ;

      [y,m] = vl_nnbnorm(x,g,b) ;
      [y_,m_] = vl_nnbnorm(x,g,b,'moments',m) ;

      test.eq(y,y_) ;
      test.eq(m,m_) ;

      dzdy = test.randn(size(y)) ;
      [dzdx,dzdg,dzdb,m__] = vl_nnbnorm(x,g,b,dzdy) ;
      [dzdx_,dzdg_,dzdb_,m___] = vl_nnbnorm(x,g,b,dzdy,'moments',m) ;
 
      test.eq(m,m__)
      test.eq(m,m___)
      test.eq(dzdx,dzdx_) ;
      test.eq(dzdg,dzdg_) ;
      test.eq(dzdb,dzdb_) ;
    end

  end
end