/** @file normalize.cpp
 ** @brief Normalization block
 ** @author Andrea Vedaldi
 **/
/*
Copyright (C) 2014 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "normalize.hpp"
#include "gpu.hpp"
#include <float.h>

/* ---------------------------------------------------------------- */
/*                                                  normalize (GPU) */
/* ---------------------------------------------------------------- */

#undef xat
#undef yat
#undef zat
#define xat(t) x[(t) * offset]
#define yat(t) y[(t) * offset]
#define zat(t) z[(t) * offset]

#define __powf powf

template<typename T> __global__
void normalize_gpu_kernel (T* normalized,
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

template<typename T>
void normalize_gpu(T* normalized,
                   T const* data,
                   size_t width,
                   size_t height,
                   size_t depth,
                   size_t num,
                   size_t normDepth,
                   T kappa, T alpha, T beta)
{
  normalize_gpu_kernel<T>
    <<< divideUpwards(width*height*num, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
    (normalized, data, width, height, depth, num, normDepth, kappa, alpha, beta) ;
}

template
void normalize_gpu<float>(float* normalized,
                          float const* data,
                          size_t width,
                          size_t height,
                          size_t depth,
                          size_t num,
                          size_t normDetph,
                          float kappa, float alpha, float beta) ;

template
void normalize_gpu<double>(double* normalized,
                           double const* data,
                           size_t width,
                           size_t height,
                           size_t depth,
                           size_t num,
                           size_t normDetph,
                           double kappa, double alpha, double beta) ;


/* ---------------------------------------------------------------- */
/*                                          normalizeBackward (gpu) */
/* ---------------------------------------------------------------- */

template<typename T> __global__
void normalizeBackward_gpu_kernel(T* normalized,
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

template<typename T>
void normalizeBackward_gpu(T* normalized,
                           T const* data,
                           T const* dzdy,
                           size_t width,
                           size_t height,
                           size_t depth,
                           size_t num,
                           size_t normDepth,
                           T kappa, T alpha, T beta)
{
  normalizeBackward_gpu_kernel<T>
  <<< divideUpwards(width*height*num, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
    (normalized, data, dzdy, width, height, depth, num, normDepth, kappa, alpha, beta) ;
}

template
void normalizeBackward_gpu<float>(float* normalized,
                                  float const* data,
                                  float const* dzdy,
                                  size_t width,
                                  size_t height,
                                  size_t depth,
                                  size_t num,
                                  size_t normDetph,
                                  float kappa, float alpha, float beta) ;

template
void normalizeBackward_gpu<double>(double* normalized,
                                   double const* data,
                                   double const* dzdy,
                                   size_t width,
                                   size_t height,
                                   size_t depth,
                                   size_t num,
                                   size_t normDetph,
                                   double kappa, double alpha, double beta) ;


