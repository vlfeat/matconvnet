function exp_sift_like()

run(fullfile(mfilename('fullpath'), '../../matlab/vlg_setup.m')) ;
run ~/src/vlfeat/toolbox/vl_setup ;

net = getDSIFTNet(5) ;

im = vl_impattern('river1') ;
im = rgb2gray(im2single(imresize(im, .25))) ;
res = tinynet(net, im) ;
descrs = reshape(res(end).x,[],128)' ;

if 0
  % test binning
  thetar = linspace(0,2*pi,100) ;
  E=[] ;
  q = 0 ;
  for theta=thetar
    [u,v]=meshgrid(1:224,1:224) ;
    im_ = single(cos(theta)*u+sin(theta)*v) ;
    res__ = tinynet(net, im_) ;
    q=q+1 ;
    for t=1:8
      E(q,t) = [res__(4).x(20,20,t)] ;
    end
    figure(2) ;clf;
    subplot(2,2,1) ;imagesc(im_) ; axis equal ;
    subplot(2,2,2) ; plot(E) ;
    subplot(2,2,3) ; vl_imarraysc(res__(3).x,'uniform',1) ; axis equal;
    subplot(2,2,4) ; vl_imarraysc(res__(4).x,'uniform',1) ; axis equal;
    drawnow ;
  end
end



