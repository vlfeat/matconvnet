// @file nnnormalize_gpu.c
// @brief Normalize block implementation (GPU)
// @author Andrea Vedaldi

/*
Copyright (C) 2014-17 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "nnnormalize.hpp"
#include "datacu.hpp"

// -------------------------------------------------------------------
//                                                             Helpers
// -------------------------------------------------------------------

#undef xat
#undef yat
#undef zat
#define xat(t) x[(t) * offset]
#define yat(t) y[(t) * offset]
#define zat(t) z[(t) * offset]

template<typename T> __global__ void
normalize_forward_kernel
(T* output,
 T const* data,
 int width,
 int height,
 int depth,
 int num,
 int normDepth,
 T kappa, T alpha, T beta)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x ;
  if (index < width*height*num) {
    int u0 = index ;
    int v0 = u0 / width ;
    int k0 = v0 / height ;
    u0 %= width ;
    v0 %= height ;

    int m1 = ((signed)normDepth-1)/2 ;
    int m2 = normDepth - m1 - 1 ;
    int offset = width*height ;
    int t ;
    T const* x = data + u0 + (v0 + k0 * (depth*height)) * width ;
    T* y = output + u0 + (v0 + k0 * (depth*height)) * width ;
    T acc = 0 ;
    for (t = -m2 ; t < (signed)depth ; ++t) {
      T ap = 0 ;
      T am = 0 ;
      if (t-m1-1 >= 0) { am = xat(t-m1-1) ; }
      if (t+m2 < depth) { ap = xat(t+m2) ; }
      acc += ap*ap - am*am ;
      if (0 <= t && t < depth) {
        yat(t) = xat(t) * powf(kappa + alpha * acc, -beta) ;
      }
    }
  }
}

template<typename T> __global__ void
normalize_backward_kernel
(T* output,
 T const* data,
 T const* dzdy,
 int width,
 int height,
 int depth,
 int num,
 int normDepth,
 T kappa, T alpha, T beta)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x ;
  if (index < width*height*num) {
    int u0 = index ;
    int v0 = u0 / width ;
    int k0 = v0 / height ;
    u0 %= width ;
    v0 %= height ;

    int m1 = ((signed)normDepth-1)/2 ;
    int m2 = normDepth - m1 - 1 ;
    int offset = width*height ;
    T ab2 = 2*alpha*beta ;
    int t, q ;
    T const* x = data + u0 + (v0 + k0 * (depth*height)) * width ;
    T* y = output + u0 + (v0 + k0 * (depth*height)) * width ;
    T const* z = dzdy + u0 + (v0 + k0 * (depth*height)) * width ;
    T acc = 0 ;
    for (t = 0 ; t < (signed)depth ; ++t) {
      yat(t) = 0 ;
    }
    for (t = -m2 ; t < (signed)depth ; ++t) {
      int q1 = t-m1 ;
      int q2 = t+m2 ;
      T ap = 0 ;
      T am = 0 ;
      if (t-m1-1 >= 0) { am = xat(t-m1-1) ; } else { q1 = 0 ; }
      if (t+m2 < depth) { ap = xat(t+m2) ; } else { q2 = depth - 1 ; }
      acc += ap*ap - am*am ;
      T L = kappa + alpha * acc ;
      T Lbeta = powf(L, -beta) ;
      T Lbeta1 = Lbeta / L ;

      if (0 <= t && t < depth) {
        yat(t) += zat(t) * Lbeta ;
        for (q = q1 ; q <= q2 ; ++ q) {
          yat(q) -= zat(t) * xat(t) * xat(q) * ab2 * Lbeta1 ;
        }
      }
    }
  }
}

// -------------------------------------------------------------------
//                                                         Forward GPU
// -------------------------------------------------------------------

template<vl::DataType dataType>
struct LRNForward<vl::VLDT_GPU, dataType>
{
  vl::ErrorCode operator()(vl::nn::LRN &op,
                           vl::Tensor &output,
                           vl::Tensor const &input)
  {
    typedef typename vl::DataTypeTraits<dataType>::type type ;
    auto width = output.getWidth() ;
    auto height = output.getHeight() ;
    auto depth = output.getDepth() ;
    auto size = output.getSize() ;
    auto inputData = (type const*)input.getMemory() ;
    auto outputData = (type*)output.getMemory() ;

    normalize_forward_kernel<type>
    <<< divideAndRoundUp(width*height*size, (size_t)VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
    (outputData, inputData, width, height, depth, size,
     op.normDepth, op.kappa, op.alpha, op.beta) ;

    cudaError_t status = cudaPeekAtLastError() ;
    return (status == cudaSuccess) ? vl::VLE_Success : vl::VLE_Cuda ;
  }
} ;

// -------------------------------------------------------------------
//                                                        Backward GPU
// -------------------------------------------------------------------

template<vl::DataType dataType>
struct LRNBackward<vl::VLDT_GPU, dataType>
{
  vl::ErrorCode operator()(vl::nn::LRN &op,
                           vl::Tensor &derInput,
                           vl::Tensor const &input,
                           vl::Tensor const &derOutput)
  {
    typedef typename vl::DataTypeTraits<dataType>::type type ;
    auto width = derOutput.getWidth() ;
    auto height = derOutput.getHeight() ;
    auto depth = derOutput.getDepth() ;
    auto size = derOutput.getSize() ;
    auto inputData = (type const*)input.getMemory() ;
    auto derOutputData = (type const*)derOutput.getMemory() ;
    auto derInputData = (type*)derInput.getMemory() ;

    normalize_backward_kernel<type >
    <<< divideAndRoundUp(width*height*size, (size_t)VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
    (derInputData, inputData, derOutputData, width, height, depth, size,
     op.normDepth, op.kappa, op.alpha, op.beta) ;

    cudaError_t status = cudaPeekAtLastError() ;
    return (status == cudaSuccess) ? vl::VLE_Success : vl::VLE_Cuda ;
  }
} ;



