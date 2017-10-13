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

template<typename T>
static inline T floor_divide(T a, T b) {
  if (a >= 0) return a/b;
  else return (a - b + 1)/b;
}

template<typename T>
static inline T ceil_divide(T a, T b) {
  if (a >= 0) return (a + b - 1)/b ;
  else return a/b ;
}

template<typename T>
static inline T static_max(T a, T b) {
  return (a>=b) ? a:b ;
}

template<typename T>
static inline T static_min(T a, T b) {
  return (a<=b) ? a:b ;
}

namespace vl { namespace impl {


  template<typename type>
  struct im2row<vl::VLDT_CPU, type>
  {

    /* ------------------------------------------------------------ */
    /*                                                      forward */
    /* ------------------------------------------------------------ */

    static vl::ErrorCode
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
            int dilateX,
            int dilateY)
    {
      auto windowExtentX = as_signed(windowWidth - 1)*dilateX + 1 ;
      auto windowExtentY = as_signed(windowHeight - 1)*dilateY + 1 ;
      auto numPatchesX = (as_signed(width + padLeft + padRight) - windowExtentX)/as_signed(strideX) + 1 ;
      auto numPatchesY = (as_signed(height + padTop + padBottom) - windowExtentY)/as_signed(strideY) + 1 ;
      auto numRows = as_signed(windowWidth * windowHeight * depth) ;

      /*
       Fill a row of the patch matrix. Since patches are stored
       along the columns of the matrix, scanning a row menas visiting all
       the patches. Different rows corresponds to a different
       offset within each patch.

       In this manner, as we fill a row
       we tend to access spatially adiacent elements
       in the input image, particulary for small strides.
       */
      for (long row = 0; row < numRows ; ++row) {
        /*
         Get the patch offset corresponding to this row of the stacked
         image.
         */
        auto u = row ;
        auto v = u / as_signed(windowWidth) ;
        auto z = v / as_signed(windowHeight) ;
        u %= windowWidth ;
        v %= windowHeight ;

        /*
         Filling this row requires visiting the pixels in the input tensor
         `data` that appear at the given offset (u,v) in the output patches.
         For the patch at (x,y), the pixel coordinates (x_data,y_data) in the
         `data` tensor are:

         x_data(x) = x * strideX + u * dilateX - padLeft,  0 <= x < numPatchesX,
         y_data(y) = y * strideY + v * dilateY - padTop,   0 <= y < numPatchesY,
         z_data(z) = z.

         Now we visit all patches (x,y) in lexicographical order to fill
         successive output pixels. Patches around the boundary may peek outside
         the `data` tensor, which is padded with zero. We calcualte these
         borders here and fill them with zeros in the output.
         
         In particular, patch x peeks within the input tensor `data`
         if x is in the range [x0,x1] given by:

         x_data(x) >= 0
         <=> x >= (padLeft - u * dilateX) / stride
         <=> x >= ceil((padLeft - u * dilateX) / stride) = x0
         
         x_data(x) <= width-1
         <=> x <= (width-1 + padLeft - u * dilateX) / stride
         <=> x <= floor((width-1 + padLeft - u * dilateX) / stride)
         <=> x <  floor((width-1 + padLeft - u * dilateX) / stride) + 1 = x1

         and the same for y. Note that, while usually x0 <= x1, there are
         special cases for which x1 < x0. This is accounted for in the loops
         below.
         */

        auto x0 = static_min(numPatchesX, ceil_divide(as_signed(padLeft) - u * dilateX, as_signed(strideX))) ;
        auto y0 = static_min(numPatchesY, ceil_divide(as_signed(padTop) - v * dilateY, as_signed(strideY))) ;
        auto x1 = static_min(numPatchesX, floor_divide(as_signed(width + padLeft) - u * dilateX - 1, as_signed(strideX)) + 1) ;
        auto y1 = static_min(numPatchesY, floor_divide(as_signed(height + padTop) - v * dilateY - 1, as_signed(strideY)) + 1) ;
        long x ;
        long y ;

        for (y = 0 ; y < y0 ; ++y) {
          for (x = 0 ; x < numPatchesX ; ++x) {
            *stacked++ = 0 ;
          }
        }
        for ( ; y < y1 ; ++y) {
          for (x = 0 ; x < x0 ; ++x) {
            *stacked++ = 0 ;
          }
          auto y_data = y * as_signed(strideY) + v * dilateY - as_signed(padTop) ;
          auto x_data = x * as_signed(strideX) + u * (signed)dilateX - as_signed(padLeft) ;
          type const * b = data + (z * as_signed(height) + y_data) * as_signed(width) + x_data ;
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
      return vl::VLE_Success ;
    }

    /* ------------------------------------------------------------ */
    /*                                                     backward */
    /* ------------------------------------------------------------ */

    static vl::ErrorCode
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
             int dilateX,
             int dilateY)
    {

      auto windowExtentX = as_signed(windowWidth - 1)*dilateX + 1 ;
      auto windowExtentY = as_signed(windowHeight - 1)*dilateY + 1 ;
      auto numPatchesX = (as_signed(width + padLeft + padRight) - windowExtentX)/as_signed(strideX) + 1 ;
      auto numPatchesY = (as_signed(height + padTop + padBottom) - windowExtentY)/as_signed(strideY) + 1 ;
      auto numRows = as_signed(windowWidth * windowHeight * depth) ;

      memset(data, 0, sizeof(type) * width * height * depth) ;

      /*
       Do the converse of im2col, still scanning rows of the stacked image.
       See comments of im2col for an explanation of the algorithm.
       */
      for (long row = 0; row < numRows ; ++row) {
        auto u = row ;
        auto v = u / as_signed(windowWidth) ;
        auto z = v / as_signed(windowHeight) ;
        u %= windowWidth ;
        v %= windowHeight ;

        auto x0 = static_min(numPatchesX, ceil_divide(as_signed(padLeft) - u * dilateX, as_signed(strideX))) ;
        auto y0 = static_min(numPatchesY, ceil_divide(as_signed(padTop) - v * dilateY, as_signed(strideY))) ;
        auto x1 = static_min(numPatchesX, floor_divide(as_signed(width + padLeft) - u * dilateX - 1, as_signed(strideX)) + 1) ;
        auto y1 = static_min(numPatchesY, floor_divide(as_signed(height + padTop) - v * dilateY - 1, as_signed(strideY)) + 1) ;

        auto y = static_max(0L, y0) ;
        stacked += numPatchesX * static_max(0L, y) ;
        for ( ; y < y1 ; ++y) {
          auto x = static_max(0L, x0) ;
          auto y_data = y * as_signed(strideY) + v * as_signed(dilateY) - as_signed(padTop) ;
          auto x_data = x * as_signed(strideX) + u * as_signed(dilateX) - as_signed(padLeft) ;
          type * b = data + (z * as_signed(height) + y_data) * as_signed(width) + x_data ;
          stacked += x ;
          for ( ; x < x1 ; ++x) {
            *b += *stacked++ ;
            b += strideX ;
          }
          stacked += numPatchesX - x ;
        }
        stacked += numPatchesX * (numPatchesY - y) ;
      }
      return vl::VLE_Success ;
    }
  } ;

} }

// Instantiations
template struct vl::impl::im2row<vl::VLDT_CPU, float> ;

#ifdef ENABLE_DOUBLE
template struct vl::impl::im2row<vl::VLDT_CPU, double> ;
#endif
