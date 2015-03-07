// @file subsampling_gpu.cu
// @brief Subsampling block implementation (GPU)
// @author Andrea Vedaldi
// @author Karel Lenc

/*
Copyright (C) 2014-15 Andrea Vedaldi and Karel Lenc.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "subsample.hpp"
#include "../datacu.hpp"
#include <assert.h>
#include <float.h>
#include <iostream>

#ifndef ENABLE_GPU
#error "subsample_gpu.cu cannot be compiled without GPU support"
#endif

using namespace vl ;

/* ---------------------------------------------------------------- */
/*                                                subsample forward */
/* ---------------------------------------------------------------- */

template<typename T> __global__ void
subsample_gpu_kernel
(T* subsampled,
 const T* data,
 const int subsampledWidth,
 const int subsampledHeight,
 const int subsampledVolume,
 const int width,
 const int height,
 const int strideX,
 const int strideY,
 const int padLeft,
 const int padTop)
{
  int subsampledIndex = threadIdx.x + blockIdx.x * blockDim.x;
  if (subsampledIndex < subsampledVolume) {
    /* subsampledIndex = x
     + y * subsampledWidth
     + z * (subsampledWidth * subsampledHeight) ;
     */
    int px = subsampledIndex ;
    int py = px / subsampledWidth ;
    int pz = py / subsampledHeight ;
    px %= subsampledWidth ;
    py %= subsampledHeight ;
    int x1 = px * strideX - padLeft ;
    int y1 = py * strideY - padTop ;
    data += pz * (width*height) ;
    T value = 0 ;
    if (x1 >= 0 && x1 < width && y1 >= 0 && y1 < height) {
      value = data[y1 * width + x1] ;
    }
    subsampled[subsampledIndex] = value ;
  }
}

template<typename T> static vl::Error
subsample_forward_gpu(Context & context,
                      T* subsampled,
                      T const* data,
                      size_t width,
                      size_t height,
                      size_t depth,
                      size_t strideX,
                      size_t strideY,
                      size_t padLeft,
                      size_t padRight,
                      size_t padTop,
                      size_t padBottom)
{
  int subsampledWidth = (width + (padLeft+padRight) - 1)/strideX + 1 ;
  int subsampledHeight = (height + (padTop+padBottom) - 1)/strideY + 1 ;
  int subsampledVolume = subsampledWidth * subsampledHeight * depth ;
  subsample_gpu_kernel<T>
  <<< divideUpwards(subsampledVolume, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
  (subsampled, data,
   subsampledWidth, subsampledHeight, subsampledVolume,
   width, height,
   strideX, strideY,
   padLeft, padTop);
  return context.setError(context.getCudaHelper().catchCudaError("subsample_backward_gpu<>: ")) ;
}

template <> vl::Error
vl::impl::subsample_forward<vl::GPU, float>(vl::Context& context,
                                            float* subsampled,
                                            float const* data,
                                            size_t height, size_t width, size_t depth,
                                            size_t strideY, size_t strideX,
                                            size_t padTop, size_t padBottom, size_t padLeft, size_t padRight)
{
  vl::Error error ;
  error = subsample_forward_gpu<float>(context,
                                       subsampled, data,
                                       height, width, depth,
                                       strideY, strideX,
                                       padTop, padBottom, padLeft, padRight) ;
  return context.passError(error, "subsample_forward<GPU,float>: ") ;
}

/* ---------------------------------------------------------------- */
/*                                          subsampleBackward (GPU) */
/* ---------------------------------------------------------------- */

template<typename T>
__global__ void subsampleBackward_gpu_kernel
(T* dzdx,
 const T* dzdy,
 const int subsampledWidth,
 const int subsampledHeight,
 const int dataVolume,
 const int width,
 const int height,
 const int strideX,
 const int strideY,
 const int padLeft,
 const int padTop)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < dataVolume) {
    int x = index ;
    int y = x / width ;
    int z = y / height ;
    x %= width ;
    y %= height ;
    dzdy += z * subsampledHeight * subsampledWidth ;
    int px = (x + padLeft) / strideX ;
    int py = (y + padTop) / strideY ;
    if (x == strideX * px - padLeft &&
        y == strideY * py - padTop) {
      dzdx[index] = dzdy[py * subsampledWidth + px] ;
    } else {
      dzdx[index] = 0 ;
    }
  }
}

template<typename T> vl::Error
subsample_backward_gpu(vl::Context& context,
                       T* dzdx,
                       T const* dzdy,
                       size_t width,
                       size_t height,
                       size_t depth,
                       size_t strideX,
                       size_t strideY,
                       size_t padLeft,
                       size_t padRight,
                       size_t padTop,
                       size_t padBottom)
{
  int subsampledWidth = (width + (padLeft+padRight) - 1)/strideX + 1 ;
  int subsampledHeight = (height + (padTop+padBottom) - 1)/strideY + 1 ;
  int nthreads = width * height * depth ;
  subsampleBackward_gpu_kernel<T>
  <<< divideUpwards(nthreads, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
  (dzdx,
   dzdy,
   subsampledWidth, subsampledHeight, nthreads,
   width, height,
   strideX, strideY,
   padLeft, padTop);
  return context.setError(context.getCudaHelper().catchCudaError("subsample_backward_gpu<>: ")) ;
}

template <> vl::Error
vl::impl::subsample_backward<vl::GPU, float>(vl::Context& context,
                                             float* derData,
                                             float const* derSubsampled,
                                             size_t height, size_t width, size_t depth,
                                             size_t strideY, size_t strideX,
                                             size_t padTop, size_t padBottom, size_t padLeft, size_t padRight)
{
  vl::Error error ;
  error = subsample_backward_gpu<float>(context,
                                        derData, derSubsampled,
                                        height, width, depth,
                                        strideY, strideX,
                                        padTop, padBottom, padLeft, padRight) ;
  return context.passError(error, "subsample_backward<GPU,float>: ") ;
}
