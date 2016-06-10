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
    dilationx = {1 2 3 4}
    dilationy = {1 2 3 4}
  end

  methods (Test)
    function dilation_correctness(test, dilationx, dilationy, stride, pad)
      x = test.randn(9,9,4,1) ;
      w = test.randn(3,3,4,1) ;

      % dilated convolution is like a regular but larger convolution
      % with 0 holes in middle of the filter. We construct the larger
      % convolution to test the dilated convolution implementation.
      
      [szY, szX, C, N] = size(w);
      new_W = test.zeros((szY-1)*dilationy + 1, (szX-1)*dilationx + 1, C, N);
      y_points = 1:dilationy:(szY-1)*dilationy+1;
      x_points = 1:dilationx:(szX-1)*dilationx+1;
      for yi = 1:numel(y_points),
        for xi = 1:numel(x_points),
          new_W(y_points(yi), x_points(xi), :, :) = w(yi, xi, :, :);
        end
      end
      y = vl_nnconv(x,new_W,[],'pad',pad,'stride',stride) ;
      in_dilation = [dilationy dilationx] ;
      y_ = vl_nnconv(x,w,[],'dilate',in_dilation,'pad',pad,'stride',stride) ;

      test.eq(y,y_) ;

      dzdy = test.randn(size(y)) ;
      [dzdx,dzdw] = vl_nnconv(x,new_W,[],dzdy,'pad',pad,'stride',stride) ;
      [dzdx_,dzdw_] = vl_nnconv(x,w,[],dzdy,'dilate',in_dilation,'pad',pad,'stride',stride) ;
      test.eq(dzdx,dzdx_);
      test.eq(dzdw(y_points, x_points, :),dzdw_);
    end
	  
	  
    function identity_filters(test)
      x = test.randn(1,1,10,4) ;
      b = test.randn(1,size(x,3)) ;
      y = vl_nnconv(x,[],b) ;
      dzdy = test.randn(size(y)) ;
      [dzdx,dzdw,dzdb] = vl_nnconv(x,[],b,dzdy) ;
      test.der(@(x) vl_nnconv(x,[],b), x, dzdy, dzdx, test.range * 1e-2) ;
      test.der(@(b) vl_nnconv(x,[],b), b, dzdy, dzdb, test.range * 1e-2) ;
    end

    function filter_shapes(test,bias,fw,fh)
      n = 3 ;
      fn = 5 ;
      depth = 10 ;
      x = test.randn(3,5,depth,n) ;
      if fh == 0 | fw == 0
        w = test.toDataType([]) ;
      else
        w = test.randn(fh,fw,depth,fn) ;
      end
      if bias
        if numel(w)==0
          b = test.randn(1,size(x,3)) ;
        else
          b = test.randn(1,fn) ;
        end
      else
        b = test.toDataType([]) ;
      end
      y = vl_nnconv(x,w,b) ;
      dzdy = test.randn(size(y)) ;
      [dzdx,dzdw,dzdb] = vl_nnconv(x,w,b,dzdy) ;
      test.der(@(x) vl_nnconv(x,w,b), x, dzdy, dzdx, test.range * 1e-2) ;
      test.der(@(w) vl_nnconv(x,w,b), w, dzdy, dzdw, test.range * 1e-2) ;
      test.der(@(b) vl_nnconv(x,w,b), b, dzdy, dzdb, test.range * 1e-2) ;
    end

    function stride_correctness(test,emptyw,stridex,stridey)
      x = test.randn(9,9,1,1) ;
      if emptyw
        w = [] ;
      else
        w = test.randn(3,3,1,1) ;
      end
      y = vl_nnconv(x,w,[]) ;
      stride = [stridey stridex] ;
      y_ = vl_nnconv(x,w,[],'stride',stride) ;
      test.eq(y(1:stridey:end,1:stridex:end,:,:),y_) ;

      dzdy = test.randn(size(y)) ;
      dzdy(setdiff(1:end, 1:stridey:end),:,:,:) = 0 ;
      dzdy(:,setdiff(1:end, 1:stridex:end),:,:) = 0 ;
      [dzdx,dzdw] = vl_nnconv(x,w,[],dzdy) ;
      [dzdx_,dzdw_] = vl_nnconv(x,w,[],dzdy(1:stridey:end,1:stridex:end,:,:),'stride',stride) ;
      test.eq(dzdx,dzdx_);
      test.eq(dzdw,dzdw_);
    end

    function pad_correctness(test, padx1, pady1, padx2, pady2)
      x = test.randn(9,9,1,1) ;
      w = test.randn(3,3,1,1) ;
      y = vl_nnconv(x,w,[]) ;
      dzdy = test.randn(size(y)) ;
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
      x = test.randn(16,15,4,2) ;
      if emptyw
        w = [] ;
        b = test.randn(4,1) ;
      else
        w = test.randn(3,3,4,5) ;
        b = test.randn(5,1) ;
      end
      y = vl_nnconv(x,w,b,'stride',stride,'pad',pad) ;
      dzdy = test.randn(size(y)) ;
      [dzdx,dzdw,dzdb] = vl_nnconv(x,w,b,dzdy,'stride',stride,'pad',pad) ;
      test.der(@(x) vl_nnconv(x,w,b,'stride',stride,'pad',pad), x, dzdy, dzdx, test.range * 1e-2) ;
      test.der(@(w) vl_nnconv(x,w,b,'stride',stride,'pad',pad), w, dzdy, dzdw, test.range * 1e-2) ;
      test.der(@(b) vl_nnconv(x,w,b,'stride',stride,'pad',pad), b, dzdy, dzdb, test.range * 1e-2) ;
    end

    function filter_groups(test,fh,fw)
      if fh == 0 | fw == 0, return ; end
      n = 3 ;
      C = 10 ;
      w_ = test.randn(fh,fw,9,3) ;
      w = cat(4, w_(:,:,1:3,1), w_(:,:,4:6,2), w_(:,:,7:9,3)) ;
      w_(:,:,4:9,1) = 0 ;
      w_(:,:,[1:3, 7:9],2) = 0 ;
      w_(:,:,1:6,3) = 0 ;
      x = test.randn(13,9,9,n) ;
      y = vl_nnconv(x,w,[]) ;
      y_ = vl_nnconv(x,w_,[]) ;
      test.eq(y,y_) ;
      dzdy = test.randn(size(y)) ;
      [dzdx,dzdw] = vl_nnconv(x,w,[],dzdy) ;
      test.der(@(x) vl_nnconv(x,w,[]), x, dzdy, dzdx, test.range * 1e-2) ;
      test.der(@(w) vl_nnconv(x,w,[]), w, dzdy, dzdw, test.range * 1e-2) ;
    end

    function test_gpu_correctnes(test)
      if ~strcmp(test.currentDevice, 'gpu'), return ; end
      opts = {...
        {'pad', [0 0 0 0], 'stride', [1 1]}, ...
        {'pad', [5 5 8 8], 'stride', [1 1]}, ...
        {'pad', [5 5 8 8], 'stride', [3 2]}} ;

      variants = {{'nocudnn'}, ...
                  {'cudnn', 'cudnnworkspacelimit', 0}, ...
                  {'cudnn', 'cudnnworkspacelimit', +inf}} ;

      fh = 11 ;
      fw = 11 ;
      fn = 10 ;
      n = 8 ;
      depth = 8 ;
      x = test.randn(128,128,depth,n) ;
      w = test.randn(fh,fw,depth,fn) ;
      b = test.randn(1,fn) ;

      for o = 1:numel(opts)
        for v = 1:numel(variants)
          %args = horzcat(variants{v}, opts{o}, {'verbose'}) ;
          args = horzcat(variants{v}, opts{o}) ;
          y = vl_nnconv(x,w,b,args{:}) ;
          dzdy = test.randn(size(y)) ;
          [dzdx,dzdw,dzdb] = vl_nnconv(x,w,b,dzdy,args{:}) ;

          dzdy_ = gather(dzdy) ;
          y_ = vl_nnconv(gather(x), gather(w), gather(b), opts{o}{:}) ;
          [dzdx_,dzdw_,dzdb_] = vl_nnconv(gather(x),gather(w),gather(b), ...
                                          gather(dzdy), opts{o}{:}) ;

          test.eq(y, y_) ;
          test.eq(dzdx, dzdx_) ;
          test.eq(dzdw, dzdw_) ;
          test.eq(dzdb, dzdb_) ;
        end
      end
    end
  end
end
