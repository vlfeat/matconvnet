function vl_test_nnlayers(gpu)

range = 100 ;

if nargin < 1, gpu = false ; end
if gpu
  grandn = @(varargin) range * gpuArray.randn(varargin{:}) ;
  grand = @(varargin) range * gpuArray.rand(varargin{:}) ;
else
  grandn = @(varargin) range * randn(varargin{:}) ;
  grand = @(varargin) range * rand(varargin{:}) ;
end

rng(1) ;

for l=5 %setdiff(1:9,6)
  switch l
    case 1
      disp('testing vl_nnsoftamxloss multiple images convolutional') ;
      C = 10 ;
      c = [7 2 1] ;
      n = 3 ;

      % compare direct and indirect composition; this cannot
      % take large ranges
      x = grand(3,4,C,n)/range + 0.001 ; % non-negative
      y = vl_nnsoftmaxloss(x,c) ;
      y_ = vl_nnloss(vl_nnsoftmax(x),c) ;
      dzdy = grandn(size(y)) ;
      dzdx = vl_nnsoftmaxloss(x,c,dzdy) ;
      dzdx_ = vl_nnsoftmax(x,vl_nnloss(vl_nnsoftmax(x),c,dzdy)) ;
      vl_testsim(y,y_,0.1) ;
      vl_testsim(dzdx, dzdx_) ;
      vl_testder(@(x) vl_nnsoftmaxloss(x,c), x, dzdy, dzdx, 1e-6) ;

      % now larger input range
      x = grand(3,4,C,n) + range * 0.001 ; % non-negative
      y = vl_nnsoftmaxloss(x,c) ;
      dzdy = grandn(size(y)) ;
      dzdx = vl_nnsoftmaxloss(x,c,dzdy) ;
      vl_testder(@(x) vl_nnsoftmaxloss(x,c), x, dzdy, dzdx, range * 1e-6) ;

    case 2
      disp('testing vl_nnloss multiple images convolutional') ;
      C = 10 ;
      c = [7 2 1] ;
      n = 3 ;
      x = grand(3,4,C,n) + 0.001 ; % non-negative
      y = vl_nnloss(x,c) ;
      dzdy = grandn(size(y)) ;
      dzdx = vl_nnloss(x,c,dzdy) ;
      vl_testder(@(x) vl_nnloss(x,c), x, dzdy, dzdx, range * 1e-8) ;

      disp('testing vl_nnloss multiple images') ;
      C = 10 ;
      c = [7 2 1] ;
      n = 3 ;
      x = grand(1,1,C,n) + 0.001 ; % non-negative
      y = vl_nnloss(x,c) ;
      dzdy = grandn(size(y)) ;
      dzdx = vl_nnloss(x,c,dzdy) ;
      vl_testder(@(x) vl_nnloss(x,c), x, dzdy, dzdx, range * 1e-8) ;

      disp('testing vl_nnloss') ;
      C = 10 ;
      c = 7 ;
      x = grand(1,1,C,1) + 0.001 ; % non-negative
      y = vl_nnloss(x,c) ;
      dzdy = grandn(size(y)) ;
      dzdx = vl_nnloss(x,c,dzdy) ;
      vl_testder(@(x) vl_nnloss(x,c), x, dzdy, dzdx, range * 1e-8) ;

    case 3
      disp('testing vl_nnsoftmax') ;
      d = 10 ;
      n = 3 ;
      delta = 1e-6 ;
      for h=1:3
        for w=1:2
          x = grandn(h,w,d,n)/range ;
          y = vl_nnsoftmax(x) ;
          dzdy = grandn(size(y)) ;
          dzdx = vl_nnsoftmax(x, dzdy) ;
          vl_testder(@(x) vl_nnsoftmax(x), x, dzdy, dzdx, 1e-2) ;
        end
      end

    case 4
      disp('testing vl_nnconv with square, non square, and fully connected filters') ;
      n = 3 ;
      fn = 5 ;
      for fw=[1 3 5 18]
        for fh=[1 2 3 9]
          w = grandn(fh,fw,10,fn,'single') ;
          b = grandn(1,fn,'single') ;
          x = grandn(9,18,10,n,'single') ;
          y = vl_nnconv(x,w,[],'verbose') ;
          dzdy = grandn(size(y),'single') ;
          [dzdx,dzdw,dzdb] = vl_nnconv(x,w,b,dzdy,'verbose') ;
          vl_testder(@(x) vl_nnconv(x,w,b), x, dzdy, dzdx, range * 1e-2) ;
          vl_testder(@(w) vl_nnconv(x,w,b), w, dzdy, dzdw, range * 1e-2) ;
          vl_testder(@(b) vl_nnconv(x,w,b), b, dzdy, dzdb, range * 1e-2) ;
        end
      end

      disp('testing vl_nnconv stride correctness') ;
      x = grandn(9,9,1,1,'single') ;
      w = grandn(3,3,1,1,'single') ;
      y = vl_nnconv(x,w,[],'verbose') ;
      y_ = vl_nnconv(x,w,[],'verbose','stride',2) ;
      vl_testsim(y(1:2:end,1:2:end,:,:),y_) ;

      dzdy = grandn(size(y),'single') ;
      dzdy(2:2:end,:,:,:) = 0 ;
      dzdy(:,2:2:end,:,:) = 0 ;
      [dzdx,dzdw] = vl_nnconv(x,w,[],dzdy,'verbose') ;
      [dzdx_,dzdw_] = vl_nnconv(x,w,[],dzdy(1:2:end,1:2:end,:,:),'verbose','stride',2) ;
      assert(all(all(gather(abs(dzdx-dzdx_)) < 1e-3))) ;

      disp('testing vl_nnconv pad correctness') ;
      y_ = vl_nnconv(x,w,[],'verbose','pad',1) ;
      vl_testsim(y_(2:end-1,2:end-1,:,:),y) ;

      dzdy = grandn(size(y),'single') ;
      [dzdx,dzdw] = vl_nnconv(x,w,[],dzdy,'verbose') ;
      [dzdx_,dzdw_] = vl_nnconv(x,w,[],padarray(dzdy,[1 1],0,'both'),'verbose','pad',1) ;
      vl_testsim(dzdx,dzdx_) ;
      vl_testsim(dzdw,dzdw_) ;

      disp('testing vl_nnconv pad and stride combo') ;
      x = grandn(16,15,4,2,'single') ;
      w = grandn(3,3,4,5,'single') ;
      b = grandn(5,1,'single') ;
      for pad=0:2
        for stride=1:4
          y = vl_nnconv(x,w,b,'verbose','stride',stride,'pad',pad) ;
          dzdy = grandn(size(y),'single') ;
          [dzdx,dzdw,dzdb] = vl_nnconv(x,w,b,dzdy,'verbose','stride',stride,'pad',pad) ;
          vl_testder(@(x) vl_nnconv(x,w,b,'stride',stride,'pad',pad), x, dzdy, dzdx, range * 1e-2) ;
          vl_testder(@(w) vl_nnconv(x,w,b,'stride',stride,'pad',pad), w, dzdy, dzdw, range * 1e-2) ;
          vl_testder(@(b) vl_nnconv(x,w,b,'stride',stride,'pad',pad), b, dzdy, dzdb, range * 1e-2) ;
        end
      end

      disp('testing vl_nnconv filter groups') ;
      n = 3 ;
      C = 10 ;
      for fw=[1 3 9]
        for fh=[4 13]
          w_ = grandn(fh,fw,9,3,'single') ;
          w = cat(4, w_(:,:,1:3,1), w_(:,:,4:6,2), w_(:,:,7:9,3)) ;
          w_(:,:,4:9,1) = 0 ;
          w_(:,:,[1:3, 7:9],2) = 0 ;
          w_(:,:,1:6,3) = 0 ;
          x = grandn(13,9,9,n,'single') ;
          y = vl_nnconv(x,w,[],'verbose') ;
          y_ = vl_nnconv(x,w_,[],'verbose') ;
          vl_testsim(y,y_) ;
          dzdy = grandn(size(y),'single') ;
          [dzdx,dzdw] = vl_nnconv(x,w,[],dzdy,'verbose') ;
          vl_testder(@(x) vl_nnconv(x,w,[]), x, dzdy, dzdx, range * 1e-2) ;
          vl_testder(@(w) vl_nnconv(x,w,[]), w, dzdy, dzdw, range * 1e-2) ;
        end
      end

    case 5
      methods = {'avg', 'max'};
      for mi = 1:numel(methods)
        fprintf('testing vl_nnpool - %s', methods{mi}) ;
        % make sure that all elements in x are different. in this way,
        % we can compute numerical derivatives reliably by adding a delta < .5.
        x = grandn(15,14,3,2,'single') ;
        x(:) = randperm(numel(x))' ;
        for pool=1:3
          for pad=0:min(3,pool-1)
            for stride=1:4
              args = {'verbose','stride',stride,'pad',pad, 'method', methods{mi}};
              y = vl_nnpool(x,pool,args{:}) ;
              dzdy = grandn(size(y),'single') ;
              dzdx = vl_nnpool(x,pool,dzdy,args{:}) ;
              vl_testder(@(x) vl_nnpool(x,pool,args{:}), ...
                         x, dzdy, dzdx, range * 1e-2) ;
            end
          end
        end

        stride = 1 ;
        pad = 0 ;
        for poolx=1:3
          for pooly=1:2
            pool = [pooly poolx] ;
            args = {'verbose','stride',stride,'pad',pad, 'method', methods{mi}};
            y = vl_nnpool(x,pool,args{:}) ;
            dzdy = grandn(size(y),'single') ;
            dzdx = vl_nnpool(x,pool,dzdy,args{:}) ;
            vl_testder(@(x) vl_nnpool(x,pool,args{:}), ...
                       x, dzdy, dzdx, range * 1e-2) ;
          end
        end

        pool = [3 2] ;
        for stridex=1:3
          for stridey=1:2
            stride = [stridey stridex] ;
            args = {'verbose','stride',stride,'pad',pad, 'method', methods{mi}};
            y = vl_nnpool(x,pool,args{:}) ;
            dzdy = grandn(size(y),'single') ;
            dzdx = vl_nnpool(x,pool,dzdy,args{:}) ;
            vl_testder(@(x) vl_nnpool(x,pool,args{:}), ...
                       x, dzdy, dzdx, range * 1e-2) ;
          end
        end

        pool = [3 4] ;
        stride = [2 1] ;
        for padLeft=0:2
          for padRight=0:2
            pad = [0 0 padLeft padRight] ;
            args = {'verbose','stride',stride,'pad',pad, 'method', methods{mi}};
            y = vl_nnpool(x,pool,args{:}) ;
            dzdy = grandn(size(y),'single') ;
            dzdx = vl_nnpool(x,pool,dzdy,args{:}) ;
            vl_testder(@(x) vl_nnpool(x,pool,args{:}), ...
                       x, dzdy, dzdx, range * 1e-2) ;
          end
        end

        pool = [3 4] ;
        stride = [2 1] ;
        for padTop=0:2
          for padBottom=0:2
            pad = [padTop padBottom 2 1] ;
            args = {'verbose','stride',stride,'pad',pad, 'method', methods{mi}};
            y = vl_nnpool(x,pool,args{:}) ;
            dzdy = grandn(size(y),'single') ;
            dzdx = vl_nnpool(x,pool,dzdy,args{:}) ;
            vl_testder(@(x) vl_nnpool(x,pool,args{:}), ...
                       x, dzdy, dzdx, range * 1e-2) ;
          end
        end
      end


    case 6
      disp('testing vl_nnnormalize') ;
      % the derivative for d=1 is not very stable numerically
      for d=2:17
        param = [d, .1, .5, .75] ;
        x = grandn(3,2,10,4,'single') ;
        y = vl_nnnormalize(x,param,'verbose') ;
        dzdy = grandn(size(y),'single') ;
        dzdx = vl_nnnormalize(x,param,dzdy,'verbose') ;
        vl_testder(@(x) vl_nnnormalize(x,param), x, dzdy, dzdx, range * 1e-3) ;
      end

      for d=1:7
        param(1) = d ;
        y = vl_nnnormalize(gather(x),param) ;
        y_ = zeros(size(y),'single') ;
        x_ = gather(x) ;
        for i=1:size(x,1)
          for j=1:size(x,2)
            for n=1:size(x,4)
              t = zeros(1,1,size(x,3),1) ;
              t(1,1,:,1) = (param(2) + param(3)*conv(squeeze(x_(i,j,:,n)).^2, ...
                ones(param(1),1), 'same')).^(-param(4)) ;
              y_(i,j,:,n) = x_(i,j,:,n) .* t ;
            end
          end
        end
        vl_testsim(y,y_) ;
      end

      x = grandn(1,1,10,1,'single') ;
      y = vl_nnnormalize(x, [20, 0, 1, .5]) ;
      vl_testsim(sum(y(:).^2), 1, 1e-2) ;

    case 7
      disp('testing relu') ;
      % make sure that all elements in x are different. in this way,
      % we can compute numerical derivatives reliably by adding a delta < .5.
      x = randn(5,5,1,1,'single') ;
      x(:) = randperm(numel(x))' - round(numel(x)/2) ;
      % avoid non-diff value for test
      x(x==0)=1 ;
      if gpu, x = gpuArray(x) ; end
      y = vl_nnrelu(x) ;
      dzdy = grandn(size(y),'single') ;
      dzdx = vl_nnrelu(x,dzdy) ;
      vl_testder(@(x) vl_nnrelu(x), x, dzdy, dzdx) ;

    case 8
       disp('testing vl_nnoffset') ;
       param = [.34, .5] ;
       x = grandn(4,5,10,3,'single') ;
       y = vl_nnoffset(x,param) ;
       dzdy = grandn(size(y),'single') ;
       dzdx = vl_nnoffset(x,param,dzdy) ;
       vl_testder(@(x) vl_nnoffset(x,param), x, dzdy, dzdx, 1e-3*range) ;

    case 9
      disp('testing vl_nndropout') ;
      x = grandn(4,5,10,3,'single') ;
      [y,mask] = vl_nndropout(x) ;
      dzdy = grandn(size(y),'single') ;
      dzdx = vl_nndropout(x,dzdy,'mask',mask) ;
      vl_testder(@(x) vl_nndropout(x,'mask',mask), x, dzdy, dzdx, 1e-3*range) ;
  end
end
