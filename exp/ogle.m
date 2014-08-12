function ogle()
run(fullfile(mfilename('fullpath'), '../../matlab/vlg_setup.m')) ;
run ~/src/vlfeat/toolbox/vl_setup ;

%load(fullfile(vlg_root, 'data', 'tinynet_caffe_data.mat')) ;

switch 2
  case 1
    net = load(fullfile(vlg_root, 'data', 'tinynet_caffe.mat')) ;
    normalize = @(x) caffe_normalize(net, x) ;
    denormalize = @(x) caffe_denormalize(net, x) ;

  case 2
    net = getDSIFTNet(5) ;
    normalize = @(x) single(rgb2gray(x)) ;
    denormalize = @(x) x ;
end

lambda = 0.0001 ;
eta = 0.01 ;
eta = 10 ;


%im = imread('http://www.mousematuk.co.uk/Mouse%20Mats/Dogs/Greyhound/slides/Greyhound%209J007D-14.JPG') ;
%im = imread('http://blog.spoongraphics.co.uk/wp-content/uploads/2010/abstract-pattern/32.jpg') ;
run ~/src/vlfeat/toolbox/vl_setup ;
im = im2uint8(vl_impattern('river1')) ;
im = imresize(im, .25) ;

for l=5;%1:numel(net.layers)
  net_ = net ;
  net_.layers = net_.layers(1:l) ;

  % get representation
  res = tinynet(net_, normalize(im)) ;
  y = res(end).x ;
  meanValue = sum(res(1).x(:)) ;

  % initial reconstruction
  recon = randn(size(normalize(im)), 'single') ;
  recon = recon + meanValue /numel(recon) ;
  %recon = recon / sum(recon(:)) * meanValue ;

  % reconstruction
  for t=1:1000
    res = tinynet(net_, recon) ;
    recony = res(end).x ;
    E(1,t) = sum((recony(:) - y(:)).^2) ;
    E(2,t) = lambda/2 * sum(recon(:).^2) ;
    E(3,t) = lambda/2 * (sum(recon(:)) - meanValue).^2 ;
    E(4,t) = E(1,t)+E(2,t)+E(3,t) ;
    dzdy = 2*(recony - y) ;

    res = tinynet(net_, recon, dzdy) ;
    dzdx = res(1).dzdx ;
    dzdx = dzdx + lambda * recon ;
    dzdx = dzdx + lambda * (sum(recon(:)) - meanValue) ;

    recon = recon - eta*dzdx ;

    if mod(t,1)==0
      figure(l) ; clf;
      subplot(2,2,1) ;
      imagesc(vl_imsc(denormalize(recon))); axis image; colormap gray;
      subplot(2,2,2) ;
      semilogy(E') ;
      legend('res', 'reg', 'meanval', 'tot') ;
      subplot(2,2,3) ;
      plot([recony(1:500); y(1:500)]') ;
      title(sprintf('output of layer %d %s', l, net.layers{l}.type)) ;
      drawnow ;
      %eta = eta/2 ;
    end
  end
end

