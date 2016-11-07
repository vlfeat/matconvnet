function varargout = vl_nnlstm(x, hp, cp, W, b, varargin)
%VL_NNLSTM
%   Implements one time-step (forward and backward) of an LSTM cell.
%
%   Reference: pg. #3 of Donahue et al.'s -- "Long-term Recurrent
%   Convolutional Networks for Visual Recognition and Description".
%
%   Note that there is no output projection step (from hn to y), which if
%   necessary should be done externally.
%
%   Inputs:
%      x:  The current input tensor of size m x N
%          m: input dimension, N: batch-size
%      hp: The previous hidden-state of size d x N
%          d: hidden/cell state dimension
%      cp: The previous cell-state of size d x N
%
%   The inputs (x, hp, cp) may also be 4D tensors with the first two
%   dimensions of size 1 (i.e., 1 x 1 x m x N). This is for added
%   compatibility with other MatConvNet layers that take 4D tensors.
%
%   Parameters:
%      W: A matrix of size: (4*d) x (m+d) = [Wxi   Whi
%         see Donahue et al. for the details of           Wxf   Whf
%         these matrices.                                 Wxo   Who
%                                                         Wxc   Whc]
%      b: Bias, a vector of size (4*d)x1
%
%   Outputs in forward-pass:
%      hn: The next hidden-state, the LSTM output (size: d x N)
%      cn: The next cell-state (size: d x N)
%
%   Input gradients, specified only in backward pass:
%      DzDhn: Gradients of loss w.r.t hn (size: d x N)
%      DzDcn: Gradients of loss w.r.t cn (size: d x N)
%
%   Outputs in backward-pass: Gradients of loss with respect to...
%      DzDx:  ...the input x (size: m x N)
%      DzDhp: ...the previous hidden state hp (size: d x N)
%      DzDcp: ...the previous cell state cp (size d x N)
%      DzDW:  ...the weight matrix W (size (4*d) x (m+d))
%      DzDb:  ...the bias vector b (size d x N)

% Copyright (C) 2016 Ankush Gupta and Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

opts.debug = false ;
[opts, grad] = vl_argparsepos(opts, varargin, 'nonrecursive') ;

x_size = size(x) ;
x = squeeze(x) ;

h_size = size(hp) ;
hp = squeeze(hp) ;
cp = squeeze(cp) ;

assert(ismatrix(x)) ;

% get batch-size and input dimension:
[m, N] = size(x) ;

% check/get sizes of W/b:
d4 = size(W,1) ;
if opts.debug
  assert(N==size(hp,2) && N==size(cp,2), 'inputs batch-size mistmatch.') ;  
  assert(mod(d4,4)==0, 'the first dimension of W should be a multiple of 4') ;
  assert(d4==size(b,1), 'size of W and b do not match.') ;
end
D = int32(d4/4) ;

if opts.debug
  assert(size(W,2)==(D+m), 'the second dimension of W should be %d.',D+m) ;
end

xh = [x; hp] ;

% do input,forget,output,g gates at once:
l_xh = bsxfun(@plus, W*xh, b) ;
l_xh(3*D+1:end,:) = 2*l_xh(3*D+1:end,:) ;
l_xh = vl_nnsigmoid(l_xh) ;
l_xh(3*D+1:end,:) = 2*l_xh(3*D+1:end,:) - 1 ; % tanh = 2sigmoid(2x)-1

ig = l_xh(1:D,:) ;
fg = l_xh(D+1:2*D,:) ;
og = l_xh(2*D+1:3*D,:) ;
gg = l_xh(3*D+1:end,:) ;

% compute the next cell and hidden states:
cn = fg.*cp + ig.*gg ;
tanh_cn = 2*vl_nnsigmoid(2*cn)-1 ;
hn = og.*tanh_cn ;

if isempty(grad) % forward-passs
  % reshape to original sizes
  hn = reshape(hn, h_size) ;
  cn = reshape(cn, h_size) ;

  varargout = {hn, cn} ;

else % backward-pass
  % extract the gradient inputs:
  [Dhn, Dcn] = grad{:} ;
  Dhn = squeeze(Dhn) ;
  Dcn = squeeze(Dcn) ;

  % wrt cn:   cn--> Dcn
  %            \__> hn -->..
  Dcn_full = Dcn + og .* Dhn .* (1.0 - tanh_cn.*tanh_cn) ;

  % wrt cp: cp --> cn
  Dcp = Dcn_full .* fg ;

  % wrt various gates:
  if isa(x, 'gpuArray')
    Dgates = gpuArray.zeros(4*D,N, classUnderlying(x)) ;
  else
    Dgates = zeros(4*D,N, 'like', x) ;
  end
  Dgates(1:D,:) = Dcn_full .* gg .* ig .* (1-ig) ; % input-gate
  Dgates(D+1:2*D,:) = Dcn_full .* cp .* fg.*(1-fg) ; % forget-gate
  Dgates(2*D+1:3*D,:) = Dhn .* tanh_cn .* og.*(1-og) ; % output-fate
  Dgates(3*D+1:end,:) = Dcn_full .* ig .* (1-gg.*gg) ;

  Dxh = W'*Dgates ;
  DW = Dgates*xh' ;
  Db = sum(Dgates,2) ;

  Dx = Dxh(1:m,:) ;
  Dhp = Dxh(m+1:end,:) ;
  
  % reshape to original sizes
  Dx = reshape(Dx, x_size) ;
  Dhp = reshape(Dhp, h_size) ;
  Dcp = reshape(Dcp, h_size) ;

  varargout = {Dx, Dhp, Dcp, DW, Db} ;
end
