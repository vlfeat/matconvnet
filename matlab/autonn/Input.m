classdef Input < Layer
  methods
    function obj = Input(name)
      if nargin >= 1
        obj.name = name ;
      end
    end
  end
end
