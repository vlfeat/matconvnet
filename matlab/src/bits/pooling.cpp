/** @file pooling.cpp
 ** @brief Max pooling filters (CPU)
 ** @author Andrea Vedaldi
 **/

#include "pooling.hpp" 
#include <algorithm>
#include <cmath>

/* ---------------------------------------------------------------- */
/*                                                 maxPooling (CPU) */
/* ---------------------------------------------------------------- */

template<typename T>
void maxPooling_cpu(T* pooled,
                    T const* data,
                    size_t width,
                    size_t height,
                    size_t depth,
                    size_t poolSize,
                    size_t stride,
                    size_t pad)
{
  int pooledWidth = (width + 2 * pad - poolSize)/stride + 1 ;
  int pooledHeight = (height + 2 * pad - poolSize)/stride + 1 ;

  for (int c = 0; c < depth; ++c) {
    for (int ph = 0; ph < pooledHeight; ++ph) {
      for (int pw = 0; pw < pooledWidth; ++pw) {
        int hstart = std::max(ph * (signed)stride - (signed)pad, 0) ;
        int wstart = std::max(pw * (signed)stride - (signed)pad, 0) ;
        int hend = std::min(hstart + poolSize, height);
        int wend = std::min(wstart + poolSize, width);
        for (int h = hstart; h < hend; ++h) {
          for (int w = wstart; w < wend; ++w) {
            pooled[ph * pooledWidth + pw] = std::max(pooled[ph * pooledWidth + pw],
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
void maxPooling_cpu<float>(float* pooled,
                           float const* data,
                           size_t width,
                           size_t height,
                           size_t depth,
                           size_t poolSize,
                           size_t stride,
                           size_t pad) ;

template
void maxPooling_cpu<double>(double* pooled,
                            double const* data,
                            size_t width,
                            size_t height,
                            size_t depth,
                            size_t poolSize,
                            size_t stride,
                            size_t pad) ;


/* ---------------------------------------------------------------- */
/*                                         maxPoolingBackward (CPU) */
/* ---------------------------------------------------------------- */

template<typename T>
void maxPoolingBackward_cpu(T* dzdx,
                            T const* data,
                            T const* dzdy,
                            size_t width,
                            size_t height,
                            size_t depth,
                            size_t poolSize,
                            size_t stride,
                            size_t pad)
{
  int pooledWidth = (width + 2*pad - poolSize)/stride + 1 ;
  int pooledHeight = (height + 2*pad - poolSize)/stride + 1 ;

  for (int c = 0; c < depth; ++c) {
    for (int ph = 0; ph < pooledHeight; ++ph) {
      for (int pw = 0; pw < pooledWidth; ++pw) {
        int hstart = std::max(ph * (signed)stride - (signed)pad, 0) ;
        int wstart = std::max(pw * (signed)stride - (signed)pad, 0) ;
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
void maxPoolingBackward_cpu<float>(float* dzdx,
                                   float const* data,
                                   float const* dzdy,
                                   size_t width,
                                   size_t height,
                                   size_t depth,
                                   size_t poolSize,
                                   size_t stride,
                                   size_t pad) ;

template
void maxPoolingBackward_cpu<double>(double* dzdx,
                                    double const* data,
                                    double const* dzdy,
                                    size_t width,
                                    size_t height,
                                    size_t depth,
                                    size_t poolSize,
                                    size_t stride,
                                    size_t pad) ;

