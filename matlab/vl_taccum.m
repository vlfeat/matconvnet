function a = vl_taccum(alpha, a, beta, b)
%VL_TACCUM  Compute A = alpha A + beta B
%   C = VL_TACCUM(ALPHA, A, BETA, B) computes efficiently
%   C = alpha A + beta B. For GPU array, it uses inplace computations.

if isscalar(a)
  a = alpha * a + beta * b ;
  return ;
elseif isa(a, 'gpuArray')
  vl_taccummex(alpha, a, beta, b, 'inplace') ;
else
  a = vl_taccummex(alpha, a, beta, b) ;
end
