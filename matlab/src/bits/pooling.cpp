/** @file pooling.cpp
 ** @brief Max pooling filters (CPU)
 ** @author Andrea Vedaldi
 ** @author Karel Lenc
 **/

/*
Copyright (C) 2014 Andrea Vedaldi and Karel Lenc.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "pooling.hpp"
#include <algorithm>
#include <cmath>
#include <cassert>

#include<iostream>

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
            int x1 = x * (signed)strideX - (signed)padLeft ;
            int y1 = y * (signed)strideY - (signed)padTop ;
            int x2 = std::min(x1 + windowWidth, width) ;
            int y2 = std::min(y1 + windowHeight, height) ;
            x1 = std::max(x1, 0) ;
            y1 = std::max(y1, 0) ;
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
            int x1 = x * (signed)strideX - (signed)padLeft ;
            int y1 = y * (signed)strideY - (signed)padTop ;
            int x2 = std::min(x1 + windowWidth, width) ;
            int y2 = std::min(y1 + windowHeight, height) ;
            x1 = std::max(x1, 0) ;
            y1 = std::max(y1, 0) ;
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
    default:
      assert(false) ;
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

/* 
 assume the output array to be cleared or otherwise
 properly initialised: accumulates the derivative
 */
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
        for (int py = 0; py < pooledHeight; ++py) {
          for (int px = 0; px < pooledWidth; ++px) {
            int x1 = px * (int)strideX - (int)padLeft ;
            int y1 = py * (int)strideY - (int)padTop ;
            int x2 = std::min(x1 + windowWidth, width) ;
            int y2 = std::min(y1 + windowHeight, height) ;
            x1 = std::max(x1, 0) ;
            y1 = std::max(y1, 0) ;
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
            dzdx[bestIndex] += dzdy[py * pooledWidth + px] ;
          }
        }
        data += width*height ;
        dzdx += width*height ;
        dzdy += pooledWidth*pooledHeight ;
      }
      break;
    case NN_POOL_AVG :
      for (int z = 0; z < depth; ++z) {
        for (int py = 0; py < pooledHeight; ++py) {
          for (int px = 0; px < pooledWidth; ++px) {
            int x1 = px * (int)strideX - (int)padLeft ;
            int y1 = py * (int)strideY - (int)padTop ;
            int x2 = std::min(x1 + windowWidth, width) ;
            int y2 = std::min(y1 + windowHeight, height) ;
            x1 = std::max(x1, 0) ;
            y1 = std::max(y1, 0) ;
            T poolSize = (y2 - y1) * (x2 - x1);
            for (int y = y1 ; y < y2 ; ++y) {
              for (int x = x1 ; x < x2 ; ++x) {
                dzdx[y * width + x] += dzdy[py * pooledWidth + px] / poolSize;
              }
            }
          }
        }
        data += width*height ;
        dzdx += width*height ;
        dzdy += pooledWidth*pooledHeight ;
      }
     break;
    default:
      assert(false) ;
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

