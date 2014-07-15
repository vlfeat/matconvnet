/** @file pooling_gpu.cu
 ** @brief Max pooling filters (GPU)
 ** @author Andrea Vedaldi
 ** @author Karel Lenc
 **/

/*
Copyright (C) 2014 Andrea Vedaldi and Karel Lenc.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "gpu.hpp"
#include "pooling.hpp"

#include <assert.h>
#include <float.h>
#include <sm_20_atomic_functions.h>

/* ---------------------------------------------------------------- */
/*                                                 maxPooling (GPU) */
/* ---------------------------------------------------------------- */

template<typename T>
__global__ void maxPooling_gpu_kernel
(T* pooled,
 const T* data,
 const int pooledWidth,
 const int pooledHeight,
 const int pooledVolume,
 const int width,
 const int height,
 const int windowWidth,
 const int windowHeight,
 const int strideX,
 const int strideY,
 const int padLeft,
 const int padTop)
{
  int pooledIndex = threadIdx.x + blockIdx.x * blockDim.x;
  if (pooledIndex < pooledVolume) {
    /* pooledIndex = x
                   + y * pooledWidth
                   + z * (pooledWidth * pooledHeight) ;
     */
    int px = pooledIndex ;
    int py = px / pooledWidth ;
    int pz = py / pooledHeight ;
    px %= pooledWidth ;
    py %= pooledHeight ;
    int x1 = px * strideX - padLeft ;
    int y1 = py * strideY - padTop ;
    int x2 = min(x1 + windowWidth, width) ;
    int y2 = min(y1 + windowHeight, height) ;
    x1 = max(x1, 0) ;
    y1 = max(y1, 0) ;
    data += pz * (width*height) ;
    T bestValue = data[y1 * width + x1] ;
    for (int y = y1 ; y < y2 ; ++y) {
      for (int x = x1 ; x < x2 ; ++x) {
        bestValue = max(bestValue, data[y * width + x]) ;
      }
    }
    pooled[pooledIndex] = bestValue ;
  }
}


template<typename T>
__global__ void avgPooling_gpu_kernel
(T* pooled,
 const T* data,
 const int pooledWidth,
 const int pooledHeight,
 const int pooledVolume,
 const int width,
 const int height,
 const int windowWidth,
 const int windowHeight,
 const int strideX,
 const int strideY,
 const int padLeft,
 const int padTop)
{
  int pooledIndex = threadIdx.x + blockIdx.x * blockDim.x;
  if (pooledIndex < pooledVolume) {
    /* pooledIndex = x
                   + y * pooledWidth
                   + z * (pooledWidth * pooledHeight) ;
     */
    int px = pooledIndex ;
    int py = px / pooledWidth ;
    int pz = py / pooledHeight ;
    px %= pooledWidth ;
    py %= pooledHeight ;
    int x1 = px * strideX - padLeft ;
    int y1 = py * strideY - padTop ;
    int x2 = min(x1 + windowWidth, width) ;
    int y2 = min(y1 + windowHeight, height) ;
    x1 = max(x1, 0) ;
    y1 = max(y1, 0) ;
    data += pz * (width*height) ;
    T accum = 0;
    T poolSize = (y2 - y1)*(x2 - x1);
    for (int y = y1 ; y < y2 ; ++y) {
      for (int x = x1 ; x < x2 ; ++x) {
        accum += data[y * width + x] ;
      }
    }
    pooled[pooledIndex] = accum / poolSize ;
  }
}


template<typename T>
void pooling_gpu(T* pooled,
                 T const* data,
                 PoolMethod method,
                 size_t width,
                 size_t height,
                 size_t depth,
                 size_t windowWidth,
                 size_t windowHeight,
                 size_t strideX,
                 size_t strideY,
                 size_t padLeft,
                 size_t padRight,
                 size_t padTop,
                 size_t padBottom)
{
  int pooledWidth = (width + (padLeft+padRight) - windowWidth)/strideX + 1 ;
  int pooledHeight = (height + (padTop+padBottom) - windowHeight)/strideY + 1 ;
  int pooledVolume = pooledWidth * pooledHeight * depth ;
  switch (method) {
    case NN_POOL_MAX :
      maxPooling_gpu_kernel<T>
      <<< divideUpwards(pooledVolume, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
      (pooled, data,
       pooledWidth, pooledHeight, pooledVolume,
       width, height,
       windowWidth, windowHeight,
       strideX, strideY,
       padLeft, padTop);
      if (cudaGetLastError() != cudaSuccess) {
        std::cout
        <<"maxPooling_gpu_kernel error ("
        <<cudaGetErrorString(cudaGetLastError())
        <<")"<<std::endl ;
      }
      break;
  case NN_POOL_AVG :
    avgPooling_gpu_kernel<T>
    <<< divideUpwards(pooledVolume, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
    (pooled, data,
     pooledWidth, pooledHeight, pooledVolume,
     width, height,
     windowWidth, windowHeight,
     strideX, strideY,
     padLeft, padTop);
    if (cudaGetLastError() != cudaSuccess) {
      std::cout
      <<"avgPooling_gpu_kernel error ("
      <<cudaGetErrorString(cudaGetLastError())
      <<")"<<std::endl ;
    }
    break;
  default:
    assert(false);
  }
}

template
void pooling_gpu<float>(float* pooled,
                        float const* data,
                        PoolMethod method,
                        size_t width,
                        size_t height,
                        size_t depth,
                        size_t windowWidth,
                        size_t windowHeight,
                        size_t strideX,
                        size_t strideY,
                        size_t padLeft,
                        size_t padRight,
                        size_t padTop,
                        size_t padBottom) ;

template
void pooling_gpu<double>(double* pooled,
                         double const* data,
                         PoolMethod method,
                         size_t width,
                         size_t height,
                         size_t depth,
                         size_t windowWidth,
                         size_t windowHeight,
                         size_t strideX,
                         size_t strideY,
                         size_t padLeft,
                         size_t padRight,
                         size_t padTop,
                         size_t padBottom) ;

/* ---------------------------------------------------------------- */
/*                                         maxPoolingBackward (GPU) */
/* ---------------------------------------------------------------- */

#ifdef VLNN_CAFFELIKE_BPPOOL
// In order to be able to use this, BP would need to have access to both
// bottom data and pooled data (currently only passed bottom data...)
template <typename T>
__global__ void maxPoolingBackward_gpu_kernel_caffelike(
    T* dzdx,
    const T* data,
    const T* pooled,
    const T* dzdy,
    const int nthreads,
    const int pooledWidth,
    const int pooledHeight,
    const int width,
    const int height,
    const int depth,
    const int windowWidth,
    const int windowHeight,
    const int strideX,
    const int strideY)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < nthreads) {
    // find out the local index
    // find out the local offset
    int x = index % width;
    int y = (index / width) % height;
    int z = (index / width / height) % depth;
    int py1 = (y < windowHeight) ? 0 : (y - windowHeight) / strideY + 1;
    int py2 = min(y / strideY + 1, pooledHeight);
    int px1 = (x < windowWidth) ? 0 : (x - windowWidth) / strideX + 1;
    int px2 = min(x / strideX + 1, pooledWidth);
    T gradient = 0;
    T datum = data[(z * height + y) * width + x];
    pooled += z * pooledHeight * pooledWidth;
    dzdy += z * pooledHeight * pooledWidth;
    for (int py = py1; py < py2; ++py) {
      for (int px = px1; px < px2; ++px) {
        gradient += dzdy[py * pooledWidth + px] *
            (datum == pooled[py * pooledWidth + px]);
      }
    }
    dzdx[index] = gradient;
  }
}
#endif


template<typename T>
__global__ void maxPoolingBackward_gpu_kernel
(T* dzdx,
 const T* data,
 const T* dzdy,
 const int pooledWidth,
 const int pooledHeight,
 const int pooledVolume,
 const int width,
 const int height,
 const int windowWidth,
 const int windowHeight,
 const int strideX,
 const int strideY,
 const int padLeft,
 const int padTop)
{
  int pooledIndex = threadIdx.x + blockIdx.x * blockDim.x;
  if (pooledIndex < pooledVolume) {
    /* pooledIndex = x
     + y * pooledWidth
     + z * (pooledWidth * pooledHeight) ;
     */
    int px = pooledIndex ;
    int py = px / pooledWidth ;
    int pz = py / pooledHeight ;
    px %= pooledWidth ;
    py %= pooledHeight ;
    data += pz * (width*height) ;
    dzdx += pz * (width*height) ;

    int x1 = px * strideX - padLeft ;
    int y1 = py * strideY - padTop ;
    int x2 = min(x1 + windowWidth, width) ;
    int y2 = min(y1 + windowHeight, height) ;
    x1 = max(x1, 0) ;
    y1 = max(y1, 0) ;
    int bestIndex = y1 * width + x1 ;
    T bestValue = data[bestIndex] ;
    for (int y = y1 ; y < y2 ; ++y) {
      for (int x = x1 ; x < x2 ; ++x) {
        int index = y * width + x ;
        T value = data[index] ;
        if (value > bestValue) {
          bestValue = value ;
          bestIndex = index ;
        }
      }
    }
    /*
     This is bad, but required to eliminate a race condition when writing
     to bottom_diff.
     Caffe goes the other way around, but requrires remembering the layer
     output, or the maximal indexes.
     atomicAdd(add, val)
     */
    atomicAdd(dzdx + bestIndex, dzdy[pooledIndex]) ;
  }
}

template <typename T>
__global__ void avgPoolingBackward_gpu_kernel(
    T* dzdx,
    const T* dzdy,
    const int nthreads,
    const int pooledWidth,
    const int pooledHeight,
    const int width,
    const int height,
    const int depth,
    const int windowWidth,
    const int windowHeight,
    const int strideX,
    const int strideY,
    const int padLeft,
    const int padTop)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < nthreads) {
    /* To understand the logic of this piece of code see the
     comments to col2im_gpu_kernel */
    int x_data = index ;
    int y_data = x_data / width ;
    int z = y_data / height ;
    x_data %= width ;
    y_data %= height ;

    int dx = x_data + padLeft - windowWidth ;
    int dy = y_data + padTop - windowHeight ;
    int px1 = (dx >= 0) ? dx/strideX + 1 : 0 ;
    int py1 = (dy >= 0) ? dy/strideY + 1 : 0 ;
    int px2 = min((x_data + padLeft) / strideX, pooledWidth - 1) ;
    int py2 = min((y_data + padTop) / strideY, pooledHeight - 1) ;
    T accumulator = 0 ;
    dzdy += z * pooledHeight * pooledWidth;
    for (int py = py1 ; py <= py2 ; ++py) {
      for (int px = px1 ; px <= px2 ; ++px) {
        int x1 = px * strideX - padLeft ;
        int y1 = py * strideY - padTop ;
        int x2 = min(x1 + windowWidth, width) ;
        int y2 = min(y1 + windowHeight, height) ;
        x1 = max(x1, 0) ;
        y1 = max(y1, 0) ;
        T poolSize = (y2 - y1) * (x2 - x1);
        accumulator += dzdy[py * pooledWidth + px] / poolSize ;
      }
    }
    dzdx[index] = accumulator ;
  }
}


template<typename T>
void poolingBackward_gpu(T* dzdx,
                         T const* data,
                         T const* dzdy,
                         PoolMethod method,
                         size_t width,
                         size_t height,
                         size_t depth,
                         size_t windowWidth,
                         size_t windowHeight,
                         size_t strideX,
                         size_t strideY,
                         size_t padLeft,
                         size_t padRight,
                         size_t padTop,
                         size_t padBottom)
{
  int pooledWidth = (width + (padLeft+padRight) - windowWidth)/strideX + 1 ;
  int pooledHeight = (height + (padTop+padBottom) - windowHeight)/strideY + 1 ;
  int nthreads;
  switch (method) {
    case NN_POOL_MAX:
      nthreads = pooledWidth * pooledHeight * depth ;
      maxPoolingBackward_gpu_kernel<T>
      <<< divideUpwards(nthreads, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
      (dzdx,
       data, dzdy,
       pooledWidth, pooledHeight, nthreads,
       width, height,
       windowWidth, windowHeight,
       strideX, strideY,
       padLeft, padTop);
      if (cudaGetLastError() != cudaSuccess) {
        std::cout
        <<"maxPooling_gpu_kernel error ("
        <<cudaGetErrorString(cudaGetLastError())
        <<")"<<std::endl ;
      }
      break;
    case NN_POOL_AVG:
      nthreads = width * height * depth ;
      avgPoolingBackward_gpu_kernel<T>
      <<< divideUpwards(nthreads, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
      (dzdx,
       dzdy,
       nthreads,
       pooledWidth,
       pooledHeight,
       width,
       height,
       depth,
       windowWidth,
       windowHeight,
       strideX,
       strideY,
       padLeft,
       padTop);
      if (cudaGetLastError() != cudaSuccess) {
        std::cout
        <<"avgPooling_gpu_kernel error ("
        <<cudaGetErrorString(cudaGetLastError())
        <<")"<<std::endl ;
      }
      break;
    default:
      assert(false) ;
  }
}

template
void poolingBackward_gpu<float>(float* dzdx,
                                float const* data,
                                float const* dzdy,
                                PoolMethod method,
                                size_t width,
                                size_t height,
                                size_t depth,
                                size_t windowWidth,
                                size_t windowHeight,
                                size_t strideX,
                                size_t strideY,
                                size_t padLeft,
                                size_t padRight,
                                size_t padTop,
                                size_t padBottom) ;

#if 0
template
void maxPoolingBackward_gpu<double>(double* dzdx,
                                    double const* data,
                                    double const* dzdy,
                                    size_t width,
                                    size_t height,
                                    size_t depth,
                                    size_t windowWidth,
                                    size_t windowHeight,
                                    size_t strideX,
                                    size_t strideY,
                                    size_t padLeft,
                                    size_t padRight,
                                    size_t padTop,
                                    size_t padBottom) ;
#endif
