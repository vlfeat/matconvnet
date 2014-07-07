function Y = vl_nnloss(X,c,dzdy)
% VL_NNLOSS  CNN log-loss
%   Y = VL_NNLOSS(X, C) computes the log-loss with repsect to class C.
%   C is a 1 x N vector with one entry per datum in the pack X.
%
%   DZDX = VL_NNLOSS(X, C, DZDY) computes the derivative DZDX of the
%   network w.r.t the input X given the derivative DZDY w.r.t.
%   the output Y.

% Copyright (C) 2014 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

% no division by zero
X = X + 1e-4 ;
c_ = c+(0:size(X,3):size(X,3)*size(X,4)-1) ;
  
if nargin <= 2
  Y = - sum(sum(sum(log(X(:,:,c_))))) ;
else
  Y_ = - (1./X) * dzdy ;
  Y = Y_*0 ;
  Y(:,:,c_) = Y_(:,:,c_) ;
end
