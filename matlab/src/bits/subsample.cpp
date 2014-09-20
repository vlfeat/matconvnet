/** @file subsample.cpp
 ** @file Subsample operator (CPU)
 ** @author Andrea Vedaldi
 ** @author Karel Lenc
 **/

/*
 Copyright (C) 2014 Andrea Vedaldi and Karel Lenc.
 All rights reserved.

 This file is part of the VLFeat library and is made available under
 the terms of the BSD license (see the COPYING file).
 */

#include "subsample.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cassert>

#include<iostream>


/* ---------------------------------------------------------------- */
/*                                                  subsample (CPU) */
/* ---------------------------------------------------------------- */

template<typename T>
void subsample_cpu(T* subsampled,
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
  int subsampledWidth = (width + (padLeft + padRight) - 1)/strideX + 1 ;
  int subsampledHeight = (height + (padTop + padBottom) - 1)/strideY + 1 ;
  for (int z = 0; z < depth; ++z) {
    for (int y = 0; y < subsampledHeight; ++y) {
      for (int x = 0; x < subsampledWidth; ++x) {
        int x1 = x * (signed)strideX - (signed)padLeft ;
        int y1 = y * (signed)strideY - (signed)padTop ;
        T value = 0 ;
        if (x1 >= 0 && x1 < width && y1 >= 0 && y1 < height) {
          value = data[y1 * width + x1] ;
        }
        subsampled[y * subsampledWidth + x] = value ;
      }
    }
    data += width*height ;
    subsampled += subsampledWidth*subsampledHeight ;
  }
}

template
void subsample_cpu<float>(float* subsampled,
                          float const* data,
                          size_t width,
                          size_t height,
                          size_t depth,
                          size_t strideX,
                          size_t strideY,
                          size_t padLeft,
                          size_t padRight,
                          size_t padTop,
                          size_t padBottom) ;

template
void subsample_cpu<double>(double* subsampled,
                           double const* data,
                           size_t width,
                           size_t height,
                           size_t depth,
                           size_t strideX,
                           size_t strideY,
                           size_t padLeft,
                           size_t padRight,
                           size_t padTop,
                           size_t padBottom) ;


/* ---------------------------------------------------------------- */
/*                                          subsampleBackward (CPU) */
/* ---------------------------------------------------------------- */

template<typename T>
void subsampleBackward_cpu(T* dzdx,
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
  int subsampledWidth = (width + (padLeft + padRight) - 1)/strideX + 1 ;
  int subsampledHeight = (height + (padTop + padBottom) - 1)/strideY + 1 ;

  memset(dzdx, 0, sizeof(T) * width * height * depth) ;

  for (int z = 0; z < depth; ++z) {
    for (int py = 0; py < subsampledHeight; ++py) {
      for (int px = 0; px < subsampledWidth; ++px) {
        int x1 = px * (int)strideX - (int)padLeft ;
        int y1 = py * (int)strideY - (int)padTop ;
        if (x1 >= 0 && x1 < width && y1 >= 0 && y1 < height) {
          dzdx[y1 * width + x1] = dzdy[py * subsampledWidth + px] ;
        }
      }
    }
    dzdx += width*height ;
    dzdy += subsampledWidth*subsampledHeight ;
  }
}

template
void subsampleBackward_cpu<float>(float* dzdx,
                                  float const* dzdy,
                                  size_t width,
                                  size_t height,
                                  size_t depth,
                                  size_t strideX,
                                  size_t strideY,
                                  size_t padLeft,
                                  size_t padRight,
                                  size_t padTop,
                                  size_t padBottom) ;

template
void subsampleBackward_cpu<double>(double* dzdx,
                                   double const* dzdy,
                                   size_t width,
                                   size_t height,
                                   size_t depth,
                                   size_t strideX,
                                   size_t strideY,
                                   size_t padLeft,
                                   size_t padRight,
                                   size_t padTop,
                                   size_t padBottom) ;

