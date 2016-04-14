function test_affineGridGenerator()
  addpath 'matlab-utils'

  insz = [128 256 3 4];
  osz = [256 512];
  gpu = [3];

  % read the images:
  i1 = imread('peppers.png');
  i2 = imread('pears.png');
  i1 = imresize(i1,insz(1:2));
  i2 = imresize(i2,insz(1:2));
 
  % prepare the input:
  x = zeros(insz,'single');
  for i = 1:(insz(4)/2)
    x(:,:,:,1 + 2*(i-1)) = i1(:,:,:);
    x(:,:,:,2 + 2*(i-1)) = i2(:,:,:);
  end
  
  % specify 4 meaninful affine transforms:
  tf1 = [1 0 0;
         0 1  0];
  tf2 = [1 0 0;
         0 1 1];
  tf3 = [1/2 0 0;
         0 1/2 0];
  th = -pi/6;
  tf4 = [cos(th)     -sin(th)  0;
         sin(th)      cos(th)  0];
  tf4(:,1:2) = tf4(:,1:2)'; % take the transpose of the rotation

  tf = {tf1,tf2,tf3,tf4};
  aff = zeros(1,1,6,4,'single');
  for i=1:4
    aff(1,1,:,i) = tf{i}(:);
  end

  if ~isempty(gpu)
    fprintf('moving to gpu\n');
    gpuDevice(gpu);
    x = gpuArray(x);
    aff = gpuArray(aff);
  end

  gridGen = dagnn.AffineGridGenerator('Ho',osz(1),'Wo',osz(2));
  %assert(osz(1)==gridGen.Ho && osz(2)==gridGen.Wo);

  % generate the affine grids:
  grids = gridGen.forward({aff},{});
  % do bilinear sampling:
  bSampler = dagnn.BilinearSampler();
  y = bSampler.forward({x, grids{1}}, {});
  y = gather(y{1});

  close all;
  clf();%figure();
  for i=1:4
    subplot(2,4,i);
    imshow(uint8(x(:,:,:,i)));
    subplot(2,4,i+4);
    imshow(uint8(y(:,:,:,i)));    
  end


  % y = uint8(gather(y{1}));
  % back-prop through the affine-grid-generator:
  dGrid = randn(size(grids{1}),'single');
  if ~isempty(gpu)
    dGrid = gpuArray(dGrid);
  end
  [dAff,~] = gridGen.backward({aff},{},{dGrid});
  dAff = gather(dAff{1});

  % check the gradients numerically:
  fprintf('===> gradient check for Aff..\n');
  fx = @(xaff) gridGen.forward({xaff},{});
  numDiffCheck(fx, aff, 20, dAff, dGrid);

  % fprintf('===> gradient check for Grid...\n');
  % %fg = @(grid) bilinear_gpu(x, grid);
  % fg = @(grid) vl_nnbilinearsampler(x,grid);
  % numDiffCheck(fg, grid, 10, ggrid, gY);

  % yu = uint8(y);
  % close all;
  % figure();
  % subplot(2,2,1);
  % imshow(i1);
  % subplot(2,2,2);
  % imshow(i2);
  % subplot(2,2,3);
  % imshow(yu(:,:,:,1));
  % subplot(2,2,4);
  % imshow(yu(:,:,:,2));
end


function d = numDiffCheck(f, x, n, din, dzdy)
  nx = numel(x);
  ep = 1e-3;
  for i=1:n
    ri = randi(nx);
    xp = x;
    xp(ri) = xp(ri) + ep;
    xn = x;
    xn(ri) = xn(ri) - ep;
    fp = f(xp);
    fp = fp{1};
    fn = f(xn);
    fn = fn{1};
    dnum = (fp - fn);
    dnum = sign(dnum) .* exp(log(abs(dnum)) - log(2*ep));
    dnum = dnum .* dzdy;
    dnum = sum(dnum(:));
    [q,w,e,r] = ind2sub(size(x),ri);
    osz = size(fn);
    fprintf('abs-diff: %.5f | dnum: %10.4f, din: %10.4f | xval: %5.2f\n', abs(dnum - din(ri)), dnum, din(ri), x(ri));
    %fprintf('\tinds: %s\n', num2str([q,w,e,r]));  
  end
end
