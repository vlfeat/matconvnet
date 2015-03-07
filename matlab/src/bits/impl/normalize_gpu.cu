// @file normalize_gpu.c
// @brief Normalize block implementation (GPU)
// @author Andrea Vedaldi

/*
Copyright (C) 2014-15 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "normalize.hpp"
#include "../datacu.hpp"
#include <assert.h>
#include <float.h>

/* ---------------------------------------------------------------- */
/*                                                normalize_forward */
/* ---------------------------------------------------------------- */

#undef xat
#undef yat
#undef zat
#define xat(t) x[(t) * offset]
#define yat(t) y[(t) * offset]
#define zat(t) z[(t) * offset]

#define __powf powf

template<typename T> __global__ void
normalize_forward_kernel
(T* normalized,
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
    T* y = normalized + u0 + (v0 + k0 * (depth*height)) * width ;
    T acc = 0 ;
    for (t = -m2 ; t < (signed)depth ; ++t) {
      T ap = 0 ;
      T am = 0 ;
      if (t-m1-1 >= 0) { am = xat(t-m1-1) ; }
      if (t+m2 < depth) { ap = xat(t+m2) ; }
      acc += ap*ap - am*am ;
      if (0 <= t && t < depth) {
        yat(t) = xat(t) * __powf(kappa + alpha * acc, -beta) ;
      }
    }
  }
}

template<> vl::Error
vl::impl::normalize_forward<vl::GPU, float>(float* normalized,
                                            float const* data,
                                            size_t width,
                                            size_t height,
                                            size_t depth,
                                            size_t size,
                                            size_t normDepth,
                                            double kappa, double alpha, double beta)
{
  normalize_forward_kernel<float>
  <<< divideUpwards(width*height*size, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
  (normalized, data, width, height, depth, size, normDepth, kappa, alpha, beta) ;

  cudaError_t status = cudaPeekAtLastError() ;
  return (status == cudaSuccess) ? vl::vlSuccess : vl::vlErrorCuda ;
}

/* ---------------------------------------------------------------- */
/*                                               normalize_backward */
/* ---------------------------------------------------------------- */

template<typename T> __global__ void
normalize_backward_kernel
(T* normalized,
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
    T* y = normalized + u0 + (v0 + k0 * (depth*height)) * width ;
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
      T Lbeta = __powf(L, -beta) ;
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

template<> vl::Error
vl::impl::normalize_backward<vl::GPU, float>(float* derData,
                                             float const* data,
                                             float const* derNormalized,
                                             size_t width,
                                             size_t height,
                                             size_t depth,
                                             size_t size,
                                             size_t normDepth,
                                             double kappa, double alpha, double beta)
{
  normalize_backward_kernel<float>
  <<< divideUpwards(width*height*size, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
  (derData, data, derNormalized, width, height, depth, size, normDepth, kappa, alpha, beta) ;

  cudaError_t status = cudaPeekAtLastError() ;
  return (status == cudaSuccess) ? vl::vlSuccess : vl::vlErrorCuda ;
}


