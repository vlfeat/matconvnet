/** @file pooling_gpu.cu
 ** @brief Max pooling filters (GPU)
 ** @author Andrea Vedaldi
 **/

/*
Copyright (C) 2014 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "gpu.hpp"
#include "pooling.hpp"

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
    int x = pooledIndex ;
    int y = x / pooledWidth ;
    int z = y / pooledHeight ;
    x %= pooledWidth ;
    y %= pooledHeight ;
    int x1 = max(x * strideX - padLeft, 0) ;
    int y1 = max(y * strideY - padTop, 0) ;
    int x2 = min(x1 + windowWidth, width) ;
    int y2 = min(y1 + windowHeight, height) ;
    data += z * (width*height) ;
    T bestValue = data[y1 * width + x1] ;
    for (int v = y1 ; v < y2 ; ++v) {
      for (int u = x1 ; u < x2 ; ++u) {
        bestValue = max(bestValue, data[v * width + u]) ;
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
    int x = pooledIndex ;
    int y = x / pooledWidth ;
    int z = y / pooledHeight ;
    x %= pooledWidth ;
    y %= pooledHeight ;
    int x1 = max(x * strideX - padLeft, 0) ;
    int y1 = max(y * strideY - padTop, 0) ;
    int x2 = min(x1 + windowWidth, width) ;
    int y2 = min(y1 + windowHeight, height) ;
    data += z * (width*height) ;
    T accum = 0;
    T poolSize = (y2 - y1)*(x2 - x1);
    for (int v = y1 ; v < y2 ; ++v) {
      for (int u = x1 ; u < x2 ; ++u) {
        accum += data[v * width + u] ;
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
    int x = pooledIndex ;
    int y = x / pooledWidth ;
    int z = y / pooledHeight ;
    x %= pooledWidth ;
    y %= pooledHeight ;

    int x1 = max(x * strideX - padLeft, 0) ;
    int y1 = max(y * strideY - padTop, 0) ;
    int x2 = min(x1 + windowWidth, width) ;
    int y2 = min(y1 + windowHeight, height) ;
    data += z * (width*height) ;
    dzdx += z * (width*height) ;
    int bestIndex = y1 * width + x1 ;
    T bestValue = data[bestIndex] ;
    for (int v = y1 ; v < y2 ; ++v) {
      for (int u = x1 ; u < x2 ; ++u) {
        int index = v * width + u ;
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

template<typename T>
__global__ void avgPoolingBackward_gpu_kernel
(T* dzdx,
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
  // TODO non trivial...
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
  int pooledVolume = pooledWidth * pooledHeight * depth ;
  switch (method) {
    case NN_POOL_MAX:
      maxPoolingBackward_gpu_kernel<T>
      <<< divideUpwards(pooledVolume, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
      (dzdx,
       data, dzdy,
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
    case NN_POOL_AVG:
      avgPoolingBackward_gpu_kernel<T>
      <<< divideUpwards(pooledVolume, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
      (dzdx,
       dzdy,
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
