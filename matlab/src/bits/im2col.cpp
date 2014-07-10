/** @file im2col.cpp
 ** @brief Image to columns and back (CPU)
 ** @author Andrea Vedaldi
 **/

/*
Copyright (C) 2014 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "im2col.hpp"
#include <string.h>

static inline int floor_divide(int a, int b) {
  if (a >= 0) return a/b;
  else return (a-b+1)/b;
}

static inline int ceil_divide(int a, int b) {
  if (a >= 0) return (a + b - 1)/b ;
  else return a/b ;
}

static inline int static_max(int a, int b) {
  return (a>=b) ? a:b ;
}

static inline int static_min(int a, int b) {
  return (a<=b) ? a:b ;
}

/* ---------------------------------------------------------------- */
/*                                                     im2col (CPU) */
/* ---------------------------------------------------------------- */

template <typename T>
void im2col_cpu(T* stacked,
                T const* data,
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
  int numPatchesX = (width + (padLeft + padRight) - windowWidth)/strideX + 1 ;
  int numPatchesY = (height + (padTop + padBottom) - windowHeight)/strideY + 1 ;
  int numRows = windowWidth * windowHeight * depth ;

  /* 
   Fill a row of the stacked image at a time. Since patches are stored
   along the columns, scanning a row menas visiting all patche once.
   Each row corresponds to a particular offset within each patch.
   
   In this manner, as we fill a row
   we tend to access spatially adiacent elements
   in the input image, particulary for small strides.
   */
  for (int row = 0; row < numRows ; ++row) {
    /* 
     Get the patch offset corresponding to this row of the stacked
     image.
     */
    int u = row ;
    int v = u / windowWidth ;
    int z = v / windowHeight ;
    u %= windowWidth ;
    v %= windowHeight ;

    /*
     Filling this row amounts to visiting all the pixels in the input
     image that appear at a given offset in the outut patches. Accounting
     for the subsampling of the output patches and input padding,
     these pixels are given by
     
     x_data(x) = x * strideX + u - padLeft,  0 <= x < numPatchesX
     y_data(y) = y * strideY + v - padTop,   0 <= y < numPatchesY
     z_data(z) = z.
     
     Here (x,y) are the spatial indexes of the output patches. Depedning
     on the padding, some of these values will read pixels outised
     the input image, which should default to 0. In particular this happens
     if
     
     x_data(x) < 0 <=> x < (padLeft - u) / stride 
                   <=> x < ceil((padLeft - u) / stride)
     x_data(x) >= width <=> x >= (width + padLeft - u) / stride
                        <=> x >= ceil((width + padLeft - u) / stride)
     
     and the same for y.
     */

    int x0 = static_max(0, ceil_divide(padLeft - u, strideX)) ;
    int y0 = static_max(0, ceil_divide(padTop - v, strideY)) ;
    int x1 = static_min(numPatchesX,  ceil_divide(width + padLeft - u, strideX)) ;
    int y1 = static_min(numPatchesY, ceil_divide(height + padTop - v, strideY)) ;

    for (int y = 0 ; y < y0 ; ++y) {
      for (int x = 0 ; x < numPatchesX ; ++x) {
        *stacked++ = 0 ;
      }
    }
    for (int y = y0 ; y < y1 ; ++y) {
      int y_data = y * strideY + v - padTop ;
      int x_data = x0 * strideX + u - padLeft ;
      T const * b = data + (z * height + y_data) * width + x_data ;

      for (int x = 0 ; x < x0 ; ++x) {
        *stacked++ = 0 ;
      }
      for (int x = x0 ; x < x1 ; ++x) {
        *stacked++ = *b ;
        b += strideX ;
      }
      for (int x = x1 ; x < numPatchesX ; ++x) {
        *stacked++ = 0 ;
      }
    }
    for (int y = y1 ; y < numPatchesY ; ++y) {
      for (int x = 0 ; x < numPatchesX ; ++x) {
        *stacked++ = 0 ;
      }
    }
  }
}

template void im2col_cpu<float>(float* stacked,
                                float const* data,
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
                                size_t padBottom);

template void im2col_cpu<double>(double* stacked,
                                 double const* data,
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
                                 size_t padBottom);

/* ---------------------------------------------------------------- */
/*                                                     col2im (CPU) */
/* ---------------------------------------------------------------- */

template <typename T>
void col2im_cpu(T* data,
                T const* stacked,
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
  int numPatchesX = (width + (padLeft + padRight) - windowWidth)/strideX + 1 ;
  int numPatchesY = (height + (padTop + padBottom) - windowHeight)/strideY + 1 ;
  int numRows = windowWidth * windowHeight * depth ;

  memset(data, 0, sizeof(T) * width * height * depth) ;

  /*
   Do the converse of im2col, still scanning rows of the stacked image.
   See comments of im2col for an explanation of the algorithms.
   */
  for (int row = 0; row < numRows ; ++row) {
    int u = row ;
    int v = u / windowWidth ;
    int z = v / windowHeight ;
    u %= windowWidth ;
    v %= windowHeight ;

    int x0 = static_max(0, ceil_divide(padLeft - u, strideX)) ;
    int y0 = static_max(0, ceil_divide(padTop - v, strideY)) ;
    int x1 = static_min(numPatchesX, ceil_divide(width + padLeft - u, strideX)) ;
    int y1 = static_min(numPatchesY, ceil_divide(height + padTop - v, strideY)) ;

    stacked += numPatchesX * y0 ;
    for (int y = y0 ; y < y1 ; ++y) {
      int y_data = y * strideY + v - padTop ;
      int x_data = x0 * strideX + u - padLeft ;
      T * b = data + (z * height + y_data) * width + x_data ;
      stacked += x0 ;
      for (int x = x0 ; x < x1 ; ++x) {
        *b += *stacked++ ;
        b += strideX ;
      }
      stacked += numPatchesX - x1 ;
    }
    stacked += numPatchesX * (numPatchesY - y1) ;
  }
}

template void col2im_cpu<float>(float* data,
                                float const* stacked,
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
                                size_t padBottom);

template void col2im_cpu<double>(double* data,
                                 double const* stacked,
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
                                 size_t padBottom);


