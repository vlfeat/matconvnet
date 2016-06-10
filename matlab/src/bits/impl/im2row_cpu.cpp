// @file im2row_cpu.cpp
// @brief Stack image patches as matrix rows (CPU)
// @author Andrea Vedaldi

/*
Copyright (C) 2014-16 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "im2row.hpp"
#include <string.h>

using namespace vl ;
using namespace vl::impl ;

/* ---------------------------------------------------------------- */
/*                                                  Heper functions */
/* ---------------------------------------------------------------- */

static inline int floor_divide(int a, int b) {
  if (a >= 0) return a/b;
  else return (a - b + 1)/b;
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

namespace vl { namespace impl {


  template<typename type>
  struct im2row<vl::CPU, type>
  {

    /* ------------------------------------------------------------ */
    /*                                                      forward */
    /* ------------------------------------------------------------ */

    static vl::Error
    forward(Context & context,
            type* stacked,
            type const* data,
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
            size_t padBottom,
            size_t dilateX,
            size_t dilateY)
    {
      int nWindowWidth = ((windowWidth - 1) * dilateX + 1);
      int nWindowHeight = ((windowHeight - 1) * dilateY + 1);
      int numPatchesX = (width + (padLeft + padRight) - nWindowWidth)/strideX + 1 ;
      int numPatchesY = (height + (padTop + padBottom) - nWindowHeight)/strideY + 1 ;
      int numRows = windowWidth * windowHeight * depth ;
      int empty_rows = ((dilateY - 1) * nWindowWidth);
      int blank_per_height = empty_rows - dilateX + 1;
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
        int nRow = ( row * dilateX ) + 
              ( row / windowWidth ) * blank_per_height -
              ( row / (windowWidth*windowHeight)) * empty_rows;
        int u = nRow ;
        int v = u / nWindowWidth ;
        int z = v / nWindowHeight ;
        u %= nWindowWidth ;
        v %= nWindowHeight ;

        /*
         Filling this row amounts to visiting all the pixels in the input
         image that appear at a given offset in the outut patches. Accounting
         for the subsampling of the output patches and input padding,
         these pixels are given by

         x_data(x) = x * strideX + u - padLeft,  0 <= x < numPatchesX
         y_data(y) = y * strideY + v - padTop,   0 <= y < numPatchesY
         z_data(z) = z.

         Here (x,y) are the spatial indexes of the output patches. Depending
         on the padding, some of these values will read pixels outised
         the input image, which should default to 0. In particular, x lands
         on a x_data(x) within the image if x0 <= x < x1 where:

         x_data(x) >= 0 <=> x >= (padLeft - u) / stride
         <=> x >= ceil((padLeft - u) / stride) = x0
         x_data(x) <= width-1 <=> x <= (width-1 + padLeft - u) / stride
         <=> x <= floor((width-1 + padLeft - u) / stride)
         <=> x <  floor((width-1 + padLeft - u) / stride) + 1 = x1

         and the same for y. Note that, while usually x0 <= x1, there are
         special cases for which x1 < x0. This is accounted for in the loops
         below.
         */

        int x0 = static_min(numPatchesX, ceil_divide(padLeft - u, strideX)) ;
        int y0 = static_min(numPatchesY, ceil_divide(padTop - v, strideY)) ;
        int x1 = static_min(numPatchesX, floor_divide(width-1 + padLeft - u, strideX) + 1) ;
        int y1 = static_min(numPatchesY, floor_divide(height-1 + padTop - v, strideY) + 1) ;
        int x ;
        int y ;

        for (y = 0 ; y < y0 ; ++y) {
          for (x = 0 ; x < numPatchesX ; ++x) {
            *stacked++ = 0 ;
          }
        }
        for ( ; y < y1 ; ++y) {
          for (x = 0 ; x < x0 ; ++x) {
            *stacked++ = 0 ;
          }
          int y_data = y * strideY + v - padTop ;
          int x_data = x * strideX + u - padLeft ;
          type const * b = data + (z * height + y_data) * width + x_data ;
          for ( ; x < x1 ; ++x) {
            *stacked++ = *b ;
            b += strideX ;
          }
          for ( ; x < numPatchesX ; ++x) {
            *stacked++ = 0 ;
          }
        }
        for ( ; y < numPatchesY ; ++y) {
          for (x = 0 ; x < numPatchesX ; ++x) {
            *stacked++ = 0 ;
          }
        }
      }
      return vl::vlSuccess ;
    }

    /* ------------------------------------------------------------ */
    /*                                                     backward */
    /* ------------------------------------------------------------ */

    static vl::Error
    backward(Context & context,
             type* data,
             type const* stacked,
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
             size_t padBottom,
             size_t dilateX,
             size_t dilateY)
    {
      int nWindowWidth = ((windowWidth - 1) * dilateX + 1);
      int nWindowHeight = ((windowHeight - 1) * dilateY + 1);
      int numPatchesX = (width + (padLeft + padRight) - nWindowWidth)/strideX + 1 ;
      int numPatchesY = (height + (padTop + padBottom) - nWindowHeight)/strideY + 1 ;
      int numRows = windowWidth * windowHeight * depth ;
      int empty_rows = ((dilateY - 1) * nWindowWidth);
      int blank_per_height = empty_rows - dilateX + 1;

      memset(data, 0, sizeof(type) * width * height * depth) ;

      /*
       Do the converse of im2col, still scanning rows of the stacked image.
       See comments of im2col for an explanation of the algorithm.
       */
      for (int row = 0; row < numRows ; ++row) {
        int nRow = ( row * dilateX ) + 
              ( row / windowWidth ) * blank_per_height -
              ( row / (windowWidth*windowHeight)) * empty_rows;
        int u = nRow ;
        int v = u / nWindowWidth ;
        int z = v / nWindowHeight ;
        u %= nWindowWidth ;
        v %= nWindowHeight ;

        int x0 = static_min(numPatchesX, ceil_divide(padLeft - u, strideX)) ;
        int y0 = static_min(numPatchesY, ceil_divide(padTop - v, strideY)) ;
        int x1 = static_min(numPatchesX, floor_divide(width-1 + padLeft - u, strideX) + 1) ;
        int y1 = static_min(numPatchesY, floor_divide(height-1 + padTop - v, strideY) + 1) ;
        int x ;
        int y ;

        y = static_max(0, y0) ;
        stacked += numPatchesX * static_max(y, 0) ;
        for ( ; y < y1 ; ++y) {
          x = static_max(0, x0) ;
          int y_data = y * strideY + v - padTop ;
          int x_data = x * strideX + u - padLeft ;
          type * b = data + (z * height + y_data) * width + x_data ;
          stacked += x ;
          for ( ; x < x1 ; ++x) {
            *b += *stacked++ ;
            b += strideX ;
          }
          stacked += numPatchesX - x ;
        }
        stacked += numPatchesX * (numPatchesY - y) ;
      }
      return vl::vlSuccess ;
    }
  } ;

} }

// Instantiations
template struct vl::impl::im2row<vl::CPU, float> ;

#ifdef ENABLE_DOUBLE
template struct vl::impl::im2row<vl::CPU, double> ;
#endif
