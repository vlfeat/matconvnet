classdef nnconvt < nntest
  properties (TestParameter)
    depth = {1 2 3}
    numImages = {1 2 3 4}
    numFilters = {1 2 3}
    upx = {1 2 3}
    upy = {1 2 3}
    padx1 = {1 2 3}
    padx2 = {1 2 3}
    pady1 = {1 2 3}
    pady2 = {1 2 3}
    up = {1 2}
    fsx = {1 2}
    crop = {1 2 3 4 5 6 7 8}
    numGroups = {1 2 3}
  end

  methods (Test)
    function basic(test, depth, numImages, numFilters)
      m = depth ;
      n = numImages ;
      k = numFilters;
      x = test.randn(10,12,m,n,'single') ;
      f = test.randn(3,4,k,m,'single') ;
      b = test.randn(1,k,'single') ;
      y = vl_nnconvt(x,f,b) ;
      dzdy = test.randn(size(y),'single') ;
      [dzdx,dzdf,dzdb] = vl_nnconvt(x,f,b,dzdy) ;
      test.der(@(x) vl_nnconvt(x,f,b), x, dzdy, dzdx, test.range * 1e-2) ;
      test.der(@(f) vl_nnconvt(x,f,b), f, dzdy, dzdf, test.range * 1e-2) ;
      test.der(@(b) vl_nnconvt(x,f,b), b, dzdy, dzdb, test.range) ;
    end

    function upsample_crop(test,upx,upy,padx1,pady1,padx2,pady2)
      m = 3 ; n = 2 ; k = 3;
      opts = {'upsample',[upy upx],'crop',[pady1 pady2 padx1 padx2]} ;
      x = test.randn(5,6,m,n,'single') ;
      f = test.randn(3,4,k,m,'single') ;
      b = test.randn(1,k,'single') ;
      y = vl_nnconvt(x,f,b,opts{:}) ;
      dzdy = test.randn(size(y),'single') ;
      [dzdx,dzdf,dzdb] = vl_nnconvt(x,f,b,dzdy,opts{:}) ;
      test.der(@(x) vl_nnconvt(x,f,b,opts{:}), x, dzdy, dzdx, test.range * 1e-2) ;
      test.der(@(f) vl_nnconvt(x,f,b,opts{:}), f, dzdy, dzdf, test.range * 1e-2) ;
      test.der(@(b) vl_nnconvt(x,f,b,opts{:}), b, dzdy, dzdb, test.range) ;
    end

    function grouped_filters(test, numGroups, depth, numFilters)
      ng = numGroups ;
      m = depth ;
      k = numFilters ;
      n = 3 ;
      opts = {'numgroups',ng} ;
      x = test.randn(10,12,m*ng,n,'single') ;
      f = test.randn(3,4,k,m*ng,'single') ;
      b = test.randn(1,k*ng,'single') ;
      y = vl_nnconvt(x,f,b,opts{:}) ;
      dzdy = test.randn(size(y),'single') ;
      [dzdx,dzdf,dzdb] = vl_nnconvt(x,f,b,dzdy,opts{:}) ;
      test.der(@(x) vl_nnconvt(x,f,b,opts{:}), x, dzdy, dzdx, test.range * 1e-2) ;
      test.der(@(f) vl_nnconvt(x,f,b,opts{:}), f, dzdy, dzdf, test.range * 1e-2) ;
      test.der(@(b) vl_nnconvt(x,f,b,opts{:}), b, dzdy, dzdb, test.range) ;
    end

    function one_one_image(test,up,fsx,crop)
      fsx = fsx*up ;
      if crop > fsx-1, return ; end
      m = 3 ;
      n = 4 ;
      k = 3 ;
      fsy = fsx * 3 ;
      x = test.randn(1,1,m,n,'single') ;
      f = test.randn(fsy,fsx,k,m,'single') ;
      b = test.randn(1,k,'single') ;
      croph = floor(crop/2) ;
      opts = {'crop', [croph, crop-croph, croph, crop-croph], 'upsample', [up up]} ;
      y = vl_nnconvt(x,f,b,opts{:}) ;
      dzdy = test.randn(size(y),'single') ;
      [dzdx,dzdf,dzdb] = vl_nnconvt(x,f,b,dzdy,opts{:}) ;
      test.der(@(x) vl_nnconvt(x,f,b,opts{:}), x, dzdy, dzdx, test.range * 1e-2) ;
      test.der(@(f) vl_nnconvt(x,f,b,opts{:}), f, dzdy, dzdf, test.range * 1e-2) ;
      test.der(@(b) vl_nnconvt(x,f,b,opts{:}), b, dzdy, dzdb, test.range * 1e-1) ;
    end
  end
end
