function removeLayer(obj, name)
%REMOVELAYER Remove a layer from the network
%   REMOVELAYER(OBJ, NAME) removes the layer NAME from the DagNN object
%   OBJ.

% Copyright (C) 2015 Karel Lenc and Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

f = find(strcmp(name, {obj.layers.name})) ;
if isempty(f), error('There is no layer ''%s''.', name), end
layer = obj.layers(f) ;
obj.layers(f) = [] ;
obj.rebuild() ;
