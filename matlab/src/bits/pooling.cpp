/** @file pooling.cpp
 ** @brief Max pooling filters (CPU)
 ** @author Andrea Vedaldi
 **/

/*
Copyright (C) 2014 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "pooling.hpp"
#include <algorithm>
#include <cmath>

/* ---------------------------------------------------------------- */
/*                                                 maxPooling (CPU) */
/* ---------------------------------------------------------------- */

template<typename T>
void pooling_cpu(T* pooled,
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
  int pooledWidth = (width + (padLeft + padRight) - windowWidth)/strideX + 1 ;
  int pooledHeight = (height + (padTop + padBottom) - windowHeight)/strideY + 1 ;

  switch (method) {
    case NN_POOL_MAX :
      for (int z = 0; z < depth; ++z) {
        for (int y = 0; y < pooledHeight; ++y) {
          for (int x = 0; x < pooledWidth; ++x) {
            int x1 = std::max(x * (signed)strideX - (signed)padLeft, 0) ;
            int y1 = std::max(y * (signed)strideY - (signed)padTop, 0) ;
            int x2 = std::min(x1 + windowWidth, width) ;
            int y2 = std::min(y1 + windowHeight, height) ;
            T bestValue = data[y1 * width + x1] ;
            for (int v = y1 ; v < y2 ; ++v) {
              for (int u = x1 ; u < x2 ; ++u) {
                bestValue = std::max(bestValue, data[v * width + u]) ;
              }
            }
            pooled[y * pooledWidth + x] = bestValue ;
          }
        }
        data += width*height ;
        pooled += pooledWidth*pooledHeight ;
      }
      break;
    case NN_POOL_AVG :
      for (int z = 0; z < depth; ++z) {
        for (int y = 0; y < pooledHeight; ++y) {
          for (int x = 0; x < pooledWidth; ++x) {
            int x1 = std::max(x * (signed)strideX - (signed)padLeft, 0) ;
            int y1 = std::max(y * (signed)strideY - (signed)padTop, 0) ;
            int x2 = std::min(x1 + windowWidth, width) ;
            int y2 = std::min(y1 + windowHeight, height) ;
            T accum = 0 ;
            T poolSize = (y2 - y1) * (x2 - x1);
            for (int v = y1 ; v < y2 ; ++v) {
              for (int u = x1 ; u < x2 ; ++u) {
                accum += data[v * width + u] ;
              }
            }
            pooled[y * pooledWidth + x] = accum / poolSize ;
          }
        }
        data += width*height ;
        pooled += pooledWidth*pooledHeight ;
      }
      break;
  }


}

template
void pooling_cpu<float>(float* pooled,
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
void pooling_cpu<double>(double* pooled,
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
/*                                         maxPoolingBackward (CPU) */
/* ---------------------------------------------------------------- */

template<typename T>
void poolingBackward_cpu(T* dzdx,
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
  int pooledWidth = (width + (padLeft + padRight) - windowWidth)/strideX + 1 ;
  int pooledHeight = (height + (padTop + padBottom) - windowHeight)/strideY + 1 ;

  switch (method) {
    case NN_POOL_MAX :
      for (int z = 0; z < depth; ++z) {
        for (int y = 0; y < pooledHeight; ++y) {
          for (int x = 0; x < pooledWidth; ++x) {
            int x1 = std::max(x * (signed)strideX - (signed)padLeft, 0) ;
            int y1 = std::max(y * (signed)strideY - (signed)padTop, 0) ;
            int x2 = std::min(x1 + windowWidth, width) ;
            int y2 = std::min(y1 + windowHeight, height) ;
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
            dzdx[bestIndex] += dzdy[y * pooledWidth + x] ;
          }
        }
        data += width*height ;
        dzdx += width*height ;
        dzdy += pooledWidth*pooledHeight ;
      }
      break;
    case NN_POOL_AVG :
      for (int z = 0; z < depth; ++z) {
        for (int y = 0; y < pooledHeight; ++y) {
          for (int x = 0; x < pooledWidth; ++x) {
            int x1 = std::max(x * (signed)strideX - (signed)padLeft, 0) ;
            int y1 = std::max(y * (signed)strideY - (signed)padTop, 0) ;
            int x2 = std::min(x1 + windowWidth, width) ;
            int y2 = std::min(y1 + windowHeight, height) ;
            T poolSize = (y2 - y1) * (x2 - x1);
            for (int v = y1 ; v < y2 ; ++v) {
              for (int u = x1 ; u < x2 ; ++u) {
                dzdx[v * width + u] += dzdy[y * pooledWidth + x] / poolSize;
              }
            }
          }
        }
        data += width*height ;
        dzdx += width*height ;
        dzdy += pooledWidth*pooledHeight ;
      }
     break;
  }
}

template
void poolingBackward_cpu<float>(float* dzdx,
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

template
void poolingBackward_cpu<double>(double* dzdx,
                                 double const* data,
                                 double const* dzdy,
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

