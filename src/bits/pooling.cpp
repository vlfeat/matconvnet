//
//  pooling.cpp
//  matconv
//
//  Created by Andrea Vedaldi on 11/02/2014.
//  Copyright (c) 2014 Andrea Vedaldi. All rights reserved.
//

#include "pooling.hpp" 


/* ---------------------------------------------------------------- */
/*                                                 maxPooling (CPU) */
/* ---------------------------------------------------------------- */

template<typename T>
void maxPooling(T* pooled,
                T const* data,
                size_t width,
                size_t height,
                size_t depth,
                size_t poolSize,
                size_t poolStride)
{
  int pooledWidth = (width - poolSize)/poolStride + 1 ;
  int pooledHeight = (height - poolSize)/poolStride + 1 ;

  for (int c = 0; c < depth; ++c) {
    for (int ph = 0; ph < pooledHeight; ++ph) {
      for (int pw = 0; pw < pooledWidth; ++pw) {
        int hstart = ph * poolStride;
        int wstart = pw * poolStride;
        int hend = std::min(hstart + poolSize, height);
        int wend = std::min(wstart + poolSize, width);
        for (int h = hstart; h < hend; ++h) {
          for (int w = wstart; w < wend; ++w) {
            pooled[ph * pooledWidth + pw] = max(pooled[ph * pooledWidth + pw],
                                                data[h * width + w]);
          }
        }
      }
    }
    data += width*height ;
    pooled += pooledWidth*pooledHeight ;
  }
}

template
void maxPooling<float>(float* pooled,
                       float const* data,
                       size_t width,
                       size_t height,
                       size_t depth,
                       size_t poolSize,
                       size_t poolStride) ;

template
void maxPooling<double>(double* pooled,
                        double const* data,
                        size_t width,
                        size_t height,
                        size_t depth,
                        size_t poolSize,
                        size_t poolStride) ;

/* ---------------------------------------------------------------- */
/*                                         maxPoolingBackward (CPU) */
/* ---------------------------------------------------------------- */

template<typename T>
void maxPoolingBackward(T* dzdx,
                        T const* data,
                        T const* dzdy,
                        size_t width,
                        size_t height,
                        size_t depth,
                        size_t poolSize,
                        size_t poolStride)
{
  int pooledWidth = (width - poolSize)/poolStride + 1 ;
  int pooledHeight = (height - poolSize)/poolStride + 1 ;

  for (int c = 0; c < depth; ++c) {
    for (int ph = 0; ph < pooledHeight; ++ph) {
      for (int pw = 0; pw < pooledWidth; ++pw) {
        int hstart = ph * poolStride;
        int wstart = pw * poolStride;
        int hend = std::min(hstart + poolSize, height);
        int wend = std::min(wstart + poolSize, width);
        int bestIndex = hstart * width + wstart ;
        T bestValue = data[bestIndex] ;
        for (int h = hstart; h < hend; ++h) {
          for (int w = wstart; w < wend; ++w) {
            int index = h * width + w ;
            T x = data[index] ;
            if (x > bestValue) {
              bestValue = x ;
              bestIndex = index ;
            }
          }
        }
        dzdx[bestIndex] += dzdy[ph * pooledWidth + pw] ;
      }
    }
    data += width*height ;
    dzdx += width*height ;
    dzdy += pooledWidth*pooledHeight ;
  }
}

template
void maxPoolingBackward<float>(float* dzdx,
                               float const* data,
                               float const* dzdy,
                               size_t width,
                               size_t height,
                               size_t depth,
                               size_t poolSize,
                               size_t poolStride) ;

template
void maxPoolingBackward<double>(double* dzdx,
                                double const* data,
                                double const* dzdy,
                                size_t width,
                                size_t height,
                                size_t depth,
                                size_t poolSize,
                                size_t poolStride) ;