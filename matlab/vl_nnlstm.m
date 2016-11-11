function varargout = vl_nnlstm(x, hp, cp, W, b, varargin)
%VL_NNLSTM Long Short-Term Memory cell.
%   [HN, CN] = VL_NNLSTM(X, HP, CP, W, B)
%   Implements one time-step of an LSTM cell.
%
%   Note that there is no output projection step, which if necessary should
%   be done externally by VL_NNCONV. VL_NNLSTM returns the so-called
%   "hidden" state directly [1].
%
%   Inputs
%   x:  The current input tensor (m x N), input dimension M, batch size N.
%   hp: The previous hidden-state (d x N), hidden/cell state dimension d.
%   cp: The previous cell-state (d x N).
%
%   W: Linear parameters matrix ((4*d) x (m+d)) = [Wxi   Whi
%      See Donahue et al. [1] for a detailed       Wxf   Whf
%      description.                                Wxo   Who
%                                                  Wxc   Whc]
%   b: Bias parameters vector ((4*d) x 1).
%
%   The inputs (x, hp, cp) may also be 4D tensors with the first two
%   dimensions of size 1 (i.e., 1 x 1 x m x N). This is for added
%   compatibility with other MatConvNet layers that take 4D tensors.
%
%   Outputs
%   hn: The next hidden-state, the LSTM output (d x N).
%   cn: The next cell-state (d x N).
%
%
%   [DZDX,DZDHP,DZDCP,DZDW,DZDB] = VL_NNLSTM(X, HP, CP, W, B, DZDHN, DZDCN)
%   Gradients of one time-step of an LSTM cell.
%
%   Inputs
%     DzDhn: Gradients of loss with respect to hn (d x N).
%     DzDcn: Gradients of loss with respect to cn (d x N).
%
%   Outputs -- gradients of loss with respect to...
%     DzDx:  ...the input x (m x N).
%     DzDhp: ...the previous hidden state hp (d x N).
%     DzDcp: ...the previous cell state cp (d x N).
%     DzDW:  ...the weight matrix W ((4*d) x (m+d)).
%     DzDb:  ...the bias vector b (d x N).
%
%
%   [...] = VL_NNLSTM(..., 'clipGrad', CLIP)
%   Applies hard gradient clipping, i.e. any gradient elements with a
%   magnitude larger than CLIP will be clamped. This only affects the
%   gradient (backward) pass. CLIP can also be a 5-elements vector, to
%   apply a different value to each of the 5 output gradient matrices.
%
%
%   See also: VL_NNLSTM_PARAMS.
%
%   Reference:
%   [1] Donahue et al., "Long-term Recurrent Convolutional Networks for
%   for Visual Recognition and Description", CVPR 2015. (pg. 3)

% Copyright (C) 2016 Ankush Gupta and Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

opts.clipGrad = 10 ;
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
assert(N==size(hp,2) && N==size(cp,2), 'Inputs/batch-size mistmatch.') ;  
assert(mod(d4,4)==0, 'The first dimension of W should be a multiple of 4') ;
assert(d4==size(b,1), 'Size of W and b do not match.') ;
D = int32(d4/4) ;

assert(size(W,2)==(D+m), 'The second dimension of W should be %d.',D+m) ;

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
  
  if ~isempty(opts.clipGrad)
    % clip gradients if necessary
    clip = opts.clipGrad ;
    if isscalar(clip)  % same clip value for all gradients
      clip = clip(ones(1, 5)) ;
    end
    for i = 1:numel(varargout)
      varargout{i} = min(clip(i), max(-clip(i), varargout{i})) ;
    end
  end
end
