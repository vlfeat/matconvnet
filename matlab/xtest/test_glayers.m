function test_glayers(gpu)

if nargin < 1, gpu = false ; end
if gpu
  grandn = @(varargin) gpuArray.randn(varargin{:}) ;
  grand = @(varargin) gpuArray.rand(varargin{:}) ;
else
  grandn = @(varargin) randn(varargin{:}) ;
  grand = @(varargin) rand(varargin{:}) ;
end

for l=1:8
  switch l
    case 1
      disp('testing gloss') ;
      n = 10 ;
      c = 10 ;
      x = grand(n,1) ; % non-negative
      y = gloss(x,c) ;
      dzdy = grandn(size(y)) ;
      dzdx = gloss(x,c,dzdy) ;
      testder(@(x) gloss(x,c), x, dzdy, dzdx) ;

    case 2
      disp('testing gsoftmax') ;
      n = 10 ;
      delta = 1e-6 ;
      x = grandn(n,1) ;
      y = gsoftmax(x) ;
      dzdy = grandn(size(y)) ;
      dzdx = gsoftmax(x, dzdy) ;
      testder(@(x) gsoftmax(x), x, dzdy, dzdx) ;

    case 3
      disp('testing gfully') ;
      m = 10;
      n = 15 ;
      x = grandn(n,1) ;
      b = grandn(m,1) ;
      w = grandn(m,n) ;
      y = gfully(x,w,b) ;
      dzdy = grandn(size(y)) ;
      [dzdx,dzdw,dzdb] = gfully(x,w,b,dzdy) ;
      testder(@(x) gfully(x,w,b), x, dzdy, dzdx) ;
      testder(@(w) gfully(x,w,b), w, dzdy, dzdw) ;
      testder(@(b) gfully(x,w,b), b, dzdy, dzdb) ;

    case 4
      disp('testing gconv stride correctness') ;
      x = grandn(9,9,1,1,'single') ;
      w = grandn(3,3,1,1,'single') ;
      y = gconv(x,w,'verbose') ;
      y_ = gconv(x,w,'verbose','stride',2) ;
      assert(isequal(y(1:2:end,1:2:end,:,:),y_)) ;

      dzdy = grandn(size(y),'single') ;
      dzdy(2:2:end,:,:,:) = 0 ;
      dzdy(:,2:2:end,:,:) = 0 ;
      [dzdw,dzdx] = gconv(x,w,dzdy,'verbose') ;
      [dzdw_,dzdx_] = gconv(x,w,dzdy(1:2:end,1:2:end,:,:),'verbose','stride',2) ;
      assert(all(all(gather(abs(dzdx-dzdx_)) < 1e-3))) ;

      disp('testing gconv pad correctness') ;
      y_ = gconv(x,w,'verbose','pad',1) ;
      assert(isequal(y_(2:end-1,2:end-1,:,:),y)) ;

      dzdy = grandn(size(y),'single') ;
      [dzdw,dzdx] = gconv(x,w,dzdy,'verbose') ;
      [dzdw_,dzdx_] = gconv(x,w,padarray(dzdy,[1 1],0,'both'),'verbose','pad',1) ;
      assert(all(all(gather(abs(dzdx-dzdx_)) < 1e-3))) ;
      assert(all(all(gather(abs(dzdw-dzdw_)) < 1e-3))) ;

      disp('testing gconv pad and stride combo') ;
      x = grandn(16,15,4,2,'single') ;
      w = grandn(3,3,4,5,'single') ;
      for pad=0:2
        for stride=1:4
          y = gconv(x,w,'verbose','stride',stride,'pad',pad) ;
          dzdy = grandn(size(y),'single') ;
          [dzdw,dzdx] = gconv(x,w,dzdy,'verbose','stride',stride,'pad',pad) ;
          testder(@(x) gconv(x,w,'stride',stride,'pad',pad), x, dzdy, dzdx, 1e-2) ;
          testder(@(w) gconv(x,w,'stride',stride,'pad',pad), w, dzdy, dzdw, 1e-2) ;
        end
      end

      disp('testing gconv filter groups') ;
      w_ = grandn(3,3,10,2,'single') ;
      w = cat(4, w_(:,:,1:5,1), w_(:,:,6:10,2)) ;
      w_(:,:,1:5,2) = 0 ;
      w_(:,:,6:10,1) = 0 ;
      x = grandn(9,9,10,1,'single') ;
      y = gconv(x,w,'verbose') ;
      y_ = gconv(x,w_,'verbose') ;
      assert(isequal(y,y_)) ;

      dzdy = grandn(size(y),'single') ;
      [dzdw,dzdx] = gconv(x,w,dzdy,'verbose') ;
      testder(@(x) gconv(x,w), x, dzdy, dzdx, 1e-2) ;
      testder(@(w) gconv(x,w), w, dzdy, dzdw, 1e-2) ;

    case 5
      disp('testing gpool') ;
      pool = [3,3] ;
      % make sure that all elements in x are different. in this way,
      % we can compute numerical derivatives reliably by adding a delta < .5.
      x = grandn(15,14,3,2,'single') ;
      x(:) = randperm(numel(x))' ;
      for pad=0:2
        for stride=1:4
          y = gpool(x,pool,'verbose','stride',stride,'pad',pad) ;
          dzdy = grandn(size(y),'single') ;
          dzdx = gpool(x,pool,dzdy,'verbose','stride',stride,'pad',pad) ;
          testder(@(x) gpool(x,pool,'stride',stride,'pad',pad), ...
            x, dzdy, dzdx, 1e-2) ;
        end
      end

    case 6
      disp('testing gnormalize') ;
      for d=1:17
        param = [d, .1, .5, .75] ;
        x = grandn(3,2,10,4,'single') ;
        y = gnormalize(x,param,'verbose') ;
        dzdy = grandn(size(y),'single') ;
        dzdx = gnormalize(x,param,dzdy,'verbose') ;
        testder(@(x) gnormalize(x,param), x, dzdy, dzdx) ;
      end

      for d=1:7
        param(1) = d ;
        y = gnormalize(gather(x),param) ;
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
        assert(all(all(all(all(gather(abs(y-y_)) < 1e-3))))) ;
      end
      
      x = grandn(1,1,10,1,'single') ;
      y = gnormalize(x, [20, 0, 1, .5]) ;
      y = gather(y) ;
      assert(abs(sum(y(:).^2)-1) < 1e-3) ;      

    case 7
      disp('testing gvec') ;
      x = grandn(3,2,10,4,'single') ;
      y = gvec(x) ;
      dzdy = grandn(size(y),'single') ;
      dzdx = gvec(x,dzdy) ;
      testder(@(x) gvec(x), x, dzdy, dzdx) ;

    case 8
      disp('testing relu') ;
      % make sure that all elements in x are different. in this way,
      % we can compute numerical derivatives reliably by adding a delta < .5.
      x = randn(5,5,1,1,'single') ;
      x(:) = randperm(numel(x))' - round(numel(x)/2) ;
      % avoid non-diff value for test
      x(x==0)=1 ;
      if gpu, x = gpuArray(x) ; end
      y = grelu(x) ;
      dzdy = grandn(size(y),'single') ;
      dzdx = grelu(x,dzdy) ;
      testder(@(x) grelu(x), x, dzdy, dzdx) ;
  end
end
