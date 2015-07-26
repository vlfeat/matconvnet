classdef nnconv < nntest
  properties (TestParameter)
    bias = {false true}
    fw = {0 1 3 5}
    fh = {0 1 2 3}
    stridex = {1 2 3}
    stridey = {1 2 3}
    emptyw = {false true}
    pad = {0 1 2}
    stride = {1 2 3 4}
    padx1 = {0 1 2}
    padx2 = {0 1 2}
    pady1 = {0 1 2}
    pady2 = {0 1 2}
  end

  methods (Test)
    function identity_filters(test)
      x = test.randn(1,1,10,4,'single') ;
      b = test.randn(1,size(x,3),'single') ;
      y = vl_nnconv(x,[],b) ;
      dzdy = test.randn(size(y),'single') ;
      [dzdx,dzdw,dzdb] = vl_nnconv(x,[],b,dzdy) ;
      test.der(@(x) vl_nnconv(x,[],b), x, dzdy, dzdx, test.range * 1e-2) ;
      test.der(@(b) vl_nnconv(x,[],b), b, dzdy, dzdb, test.range * 1e-2) ;
    end

    function filter_shapes(test,bias,fw,fh)
      n = 3 ;
      fn = 5 ;
      depth = 10 ;
      x = test.randn(3,5,depth,n,'single') ;
      if fh == 0 | fw == 0
        w = single([]) ;
      else
        w = test.randn(fh,fw,depth,fn,'single') ;
      end

      if bias
        if numel(w)==0
          b = test.randn(1,size(x,3),'single') ;
        else
          b = test.randn(1,fn,'single') ;
        end
      else
        b = single([]) ;
      end
      y = vl_nnconv(x,w,b) ;
      dzdy = test.randn(size(y),'single') ;
      [dzdx,dzdw,dzdb] = vl_nnconv(x,w,b,dzdy) ;
      test.der(@(x) vl_nnconv(x,w,b), x, dzdy, dzdx, test.range * 1e-2) ;
      test.der(@(w) vl_nnconv(x,w,b), w, dzdy, dzdw, test.range * 1e-2) ;
      test.der(@(b) vl_nnconv(x,w,b), b, dzdy, dzdb, test.range * 1e-2) ;
    end

    function stride_correctness(test,emptyw,stridex,stridey)
      x = test.randn(9,9,1,1,'single') ;
      if emptyw
        w = [] ;
      else
        w = test.randn(3,3,1,1,'single') ;
      end
      y = vl_nnconv(x,w,[]) ;
      stride = [stridey stridex] ;
      y_ = vl_nnconv(x,w,[],'stride',stride) ;
      test.eq(y(1:stridey:end,1:stridex:end,:,:),y_) ;

      dzdy = test.randn(size(y),'single') ;
      dzdy(setdiff(1:end, 1:stridey:end),:,:,:) = 0 ;
      dzdy(:,setdiff(1:end, 1:stridex:end),:,:) = 0 ;
      [dzdx,dzdw] = vl_nnconv(x,w,[],dzdy) ;
      [dzdx_,dzdw_] = vl_nnconv(x,w,[],dzdy(1:stridey:end,1:stridex:end,:,:),'stride',stride) ;
      test.eq(dzdx,dzdx_);
      test.eq(dzdw,dzdw_);
    end

    function pad_correctness(test, padx1, pady1, padx2, pady2)
      x = test.randn(9,9,1,1,'single') ;
      w = test.randn(3,3,1,1,'single') ;
      y = vl_nnconv(x,w,[]) ;
      dzdy = test.randn(size(y),'single') ;
      [dzdx,dzdw] = vl_nnconv(x,w,[],dzdy) ;

      pad = [pady1 pady2 padx1 padx2] ;
      y_ = vl_nnconv(x,w,[],'pad',pad) ;
      test.eq(y_(pady1+1:end-pady2,padx1+1:end-padx2,:,:),y) ;
      dzdy_ = padarray(padarray(dzdy,[pady1 padx1],'pre'),...
                       [pady2 padx2], 'post') ;

      [dzdx_,dzdw_] = vl_nnconv(x,w,[],dzdy_,'pad',pad) ;
      test.eq(dzdx,dzdx_) ;
      test.eq(dzdw,dzdw_) ;
    end

    function pad_and_stride(test,emptyw,pad,stride)
      x = test.randn(16,15,4,2,'single') ;
      if emptyw
        w = [] ;
        b = test.randn(4,1,'single') ;
      else
        w = test.randn(3,3,4,5,'single') ;
        b = test.randn(5,1,'single') ;
      end
      y = vl_nnconv(x,w,b,'stride',stride,'pad',pad) ;
      dzdy = test.randn(size(y),'single') ;
      [dzdx,dzdw,dzdb] = vl_nnconv(x,w,b,dzdy,'stride',stride,'pad',pad) ;
      test.der(@(x) vl_nnconv(x,w,b,'stride',stride,'pad',pad), x, dzdy, dzdx, test.range * 1e-2) ;
      test.der(@(w) vl_nnconv(x,w,b,'stride',stride,'pad',pad), w, dzdy, dzdw, test.range * 1e-2) ;
      test.der(@(b) vl_nnconv(x,w,b,'stride',stride,'pad',pad), b, dzdy, dzdb, test.range * 1e-2) ;
    end

    function filter_groups(test,fh,fw)
      if fh == 0 | fw == 0, return ; end
      n = 3 ;
      C = 10 ;
      w_ = test.randn(fh,fw,9,3,'single') ;
      w = cat(4, w_(:,:,1:3,1), w_(:,:,4:6,2), w_(:,:,7:9,3)) ;
      w_(:,:,4:9,1) = 0 ;
      w_(:,:,[1:3, 7:9],2) = 0 ;
      w_(:,:,1:6,3) = 0 ;
      x = test.randn(13,9,9,n,'single') ;
      y = vl_nnconv(x,w,[]) ;
      y_ = vl_nnconv(x,w_,[]) ;
      test.eq(y,y_) ;
      dzdy = test.randn(size(y),'single') ;
      [dzdx,dzdw] = vl_nnconv(x,w,[],dzdy) ;
      test.der(@(x) vl_nnconv(x,w,[]), x, dzdy, dzdx, test.range * 1e-2) ;
      test.der(@(w) vl_nnconv(x,w,[]), w, dzdy, dzdw, test.range * 1e-2) ;
    end
  end
end
