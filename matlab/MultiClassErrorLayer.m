classdef MultiClassErrorLayer < GenericLayer
  properties
    TopNs;
  end
  
  methods
    function obj = MultiClassErrorLayer(topNs, varargin)
      obj.TopNs = topNs;
      obj = obj.argparse(varargin);
    end
  end
  
  methods (Static)
    function top_data = forward(obj, bottom_data, top_data )
      % TODO divide the err by the number of images in a batch here!
      x = bottom_data(1).x; % Predictions
      c = bottom_data(2).x; % GT
      sz = arrayfun(@(d) size(x, d), 1:4);
      assert(sz(3) > max(obj.TopNs));
      if size(c, 2) == sz(4), c = permute(c, [3 4 1 2]); end
      c_sz = arrayfun(@(d) size(c, d), 1:4);
      assert(all(sz([1 2 4]) == c_sz([1 2 4])));
      
      n = prod(sz(1:2)) ;
      [~,x] = sort(x, 3, 'descend') ;
      error_p = gather(~bsxfun(@eq, x, c));
      error = zeros(numel(obj.TopNs), 1);
      for tni = 1:numel(obj.TopNs)
        m_err = min(error_p(:,:,1:obj.TopNs(tni),:), [], 3);
        error(tni) = sum(m_err(:));
      end
      
      top_data.x = error ./ n;
    end
    
    function bottom_data = backward(obj, bottom_data, top_data )
      % Do nothing..
    end
  end
  
end

