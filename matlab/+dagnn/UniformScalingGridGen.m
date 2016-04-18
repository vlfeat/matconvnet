% X,Y sampling grid generator for isotropic scaling transform:
%
%   INPUTS:
%       (1) transforms : 1x1x3xN tensor of affine tranform parms
%                        the three parameters are: [s cx cy]:
%                        corresponding to the following Affine transform:
%                          [   s   0   c_x ]
%                          [   0   s   c_y ]
%
%   PARAMS:
%     Ho,Wo : (scalars) spatial dimensions of the output grid
%
%   OUTPUTS:
%       (1) grid : HoxWox2xN grid of sampling coordinates,
%                  which corresponds to applying the above affine 
%                  transforms to a [-1,1]x[-1,1] output grid of size HoxWo
%
% (c) 2016 Ankush Gupta

classdef UniformScalingGridGen < dagnn.Layer

 properties
     Ho = 0;
     Wo = 0;
 end

  properties (Transient)
    % the grid --> this is cached
    % has the size: [HoWo x 2]
    xxyy ;
  end

  methods

    function outputs = forward(obj, inputs, ~)
      % input is a 1x1x3xN TENSOR corresponding to:
      % [ c1  0 c2 ]
      % [  0 c1 c3 ]
      % 
      % OUTPUT is a HoxWox2xN grid
      %fprintf('restricted-affineGridGenerator forward\n');
      % reshape the tfm params into matrices:
      T = inputs{1};
      % check shape:
      sz_T = size(T);
      assert(all(sz_T(1:3) == [1 1 3]), 'transforms have incorrect shape');
      % use gpu?:
      useGPU = isa(T, 'gpuArray');
      nbatch = size(T,4);
      S = reshape(T(1,1,1,:), 1,1,nbatch); % x,y scaling
      t = reshape(T(1,1,2:3,:), 1,2,nbatch); % translation
      % generate the grid coordinates:
      if isempty(obj.xxyy)
        obj.init_grid(useGPU);
      end
      % transform the grid:
      g = bsxfun(@times, obj.xxyy, S); % scale
      g = bsxfun(@plus, g, t); % translate
      g = reshape(g, obj.Ho,obj.Wo,2,nbatch);
      outputs = {g};
    end

    function [derInputs, derParams] = backward(obj, inputs, ~, derOutputs)
      %fprintf('restricted-affineGridGenerator backward\n');
      dY = derOutputs{1};
      useGPU = isa(dY, 'gpuArray');
      nbatch = size(dY,4);

      % create the gradient buffer:
      dA = zeros([1,1,3,nbatch], 'single');
      if useGPU, dA = gpuArray(dA); end

      dY  = reshape(dY, obj.Ho*obj.Wo,2,nbatch);
      % gradient wrt the linear part:
      dA(1,1,1,:) = reshape(obj.xxyy,1,[]) * reshape(dY, [],nbatch);
      % gradient wrt translation (or bias):
      dA(1,1,2:3,:) = sum(dY,1);

      derInputs = {dA};
      derParams = {};
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      nBatch = inputSizes{1}(4);
      outputSizes = {[obj.Ho, obj.Wo, 2, nBatch]};
    end

    function obj = UniformScalingGridGen(varargin)
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
