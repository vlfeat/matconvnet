% Affine Transform Block:
% Turns 1 x 1 x 6 x N affine transforms to: Ho x Wo x 2 x N
% sampling grid. This can be used as an example to define
% other transforms like thin-plate-splines etc. 
%
% (c) 2016 Ankush Gupta

classdef AffineGridGenerator < dagnn.Layer

 properties
     f_downsample = 1;
     Hi = 0;
     Wi = 0;
     Ho = 0;
     Wo = 0;
 end

  properties (Transient)
    % the grid (normalized \in [-1,1]) --> this is cached
    % has the size: [HoWo x 2]
    xxyy ;
  end

  methods
    function outputs = forward(obj, inputs, ~)
      % input is a 1x1x6xN TENSOR corresponding to:
      % [ c1 c2 c5 ]
      % [ c3 c4 c6 ]
      % [  0  0  1 ]
      % i.e., [x_out] = [c1 c2]  * [x_in] + [c5]
      %       [y_out]   [c3 c4]    [y_in]   [c6]
      %
      % OUTPUT is a HoxWox2xN grid which corresponds to applying 
      % the above affine transform to the [-1,1] normalized x,y
      % coordinates.

      %fprintf('affineGridGenerator forward\n');
      useGPU = isa(inputs{1}, 'gpuArray');
      
      % reshape the tfm params into matrices:
      A = inputs{1};
      nbatch = size(A,4);
      A = reshape(A, 2,3,nbatch);
      L = A(:,1:2,:);
      L = reshape(L,2,2*nbatch); % linear part

      % generate the grid coordinates:
      if isempty(obj.xxyy)
        obj.init_grid(useGPU);
      end

      % transform the grid:
      t = A(:,3,:); % translation
      t = reshape(t,1,2*nbatch);
      g = bsxfun(@plus, obj.xxyy * L, t); % apply the transform
      g = reshape(g, obj.Ho,obj.Wo,2,nbatch);

      outputs = {g};
    end

    function [derInputs, derParams] = backward(obj, inputs, ~, derOutputs)
      %fprintf('affineGridGenerator backward\n');
      
      useGPU = isa(derOutputs{1}, 'gpuArray');
      dY = derOutputs{1};
      nbatch = size(dY,4);

      % create the gradient buffer:
      dA = zeros([2,3,nbatch], 'single');
      if useGPU, dA = gpuArray(dA); end

      dY = reshape(dY, obj.Ho*obj.Wo, 2*nbatch);
      % gradient wrt the linear part:
      dL = obj.xxyy' * dY;
      dL = reshape(dL,2,2,nbatch);
      dA(:,1:2,:) = dL;

      % gradient wrt translation (or bias):
      dt = reshape(sum(dY,1),2,1,nbatch);
      dA(:,3,:) = dt;

      dA = reshape(dA, size(inputs{1}));
      derInputs = {dA};
      derParams = {};
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      nBatch = inputSizes{1}(4);
      outputSizes = {[obj.Ho, obj.Wo, 2, nBatch]};
    end

    function obj = AffineGridGenerator(varargin)
      obj.load(varargin);
      % get the output sizes:
      obj.Ho = obj.Ho;
      obj.Wo = obj.Wo;
      obj.xxyy = [];
    end

    function init_grid(obj, useGPU)
      % initialize the grid: 
      % this is a constant
      xi = linspace(-1, 1, obj.Wo);
      yi = linspace(-1, 1, obj.Ho);

      [yy,xx] = meshgrid(xi,yi);
      xxyy = [xx(:) yy(:)]; % Mx2
      if useGPU
        xxyy = gpuArray(xxyy);
      end
      obj.xxyy = xxyy; % cache it here
    end

  end
end
