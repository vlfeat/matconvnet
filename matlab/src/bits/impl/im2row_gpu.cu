// @file im2row_gpu.cu
// @brief Stack image patches as matrix rows (GPU)
// @author Andrea Vedaldi

/*
Copyright (C) 2014-15 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "im2row.hpp"
#include "../datacu.hpp"
#include <iostream>

using namespace vl ;

/* ---------------------------------------------------------------- */
/*                                                           im2row */
/* ---------------------------------------------------------------- */

template <typename T> __global__ void
im2row_forward_kernel(T* stacked,
                      T const* data,
                      const int numPatchesX,
                      const int numPatchesY,
                      const int numPatchSlices,
                      const int width,
                      const int height,
                      const int windowWidth,
                      const int windowHeight,
                      const int strideX,
                      const int strideY,
                      const int padLeft,
                      const int padTop,
                      const int dilateX,
                      const int dilateY)
{
  /* each kernel copies the pixels in an image patch for one channel */
  int index = threadIdx.x + blockIdx.x * blockDim.x ;
  int nWindowWidth = ((windowWidth - 1) * dilateX + 1);
  int nWindowHeight = ((windowHeight - 1) * dilateY + 1);
  if (index < numPatchSlices) {
    /*
     get the patch slice (x,y,z) to copy
     */
    int x = index ;
    int y = x / numPatchesX ;
    int z = y / numPatchesY ;
    x %= numPatchesX ;
    y %= numPatchesY ;

    /*
     pick the top-left corer of the patch slice in the input image
     */
    int x_data = x * strideX - padLeft ;
    int y_data = y * strideY - padTop ;
    data += (z * height + y_data) * width + x_data ;

    /*
     pick the column of the stacked image which contains this patch,
     and move down along the column at the beginning of the patch slice
     */
    int patchSliceOffset = (windowWidth*windowHeight) * z ;
    stacked += (numPatchesY * patchSliceOffset + y) * numPatchesX + x ;

    /*
     copy the patch slice
     */
    for (int v = 0 ; v < nWindowHeight ; v+=dilateY) {
      for (int u = 0 ; u < nWindowWidth ; u+=dilateX) {
        if (y_data + v >= 0 &&
            y_data + v < height &&
            x_data + u >= 0 &&
            x_data + u < width) {
          *stacked = data[v * width + u] ;
        } else {
          *stacked = 0 ;
        }
        stacked += (numPatchesX*numPatchesY) ;
      }
    }
  }
}


/* ---------------------------------------------------------------- */
/*                          im2row backward kernel without dilation */
/* ---------------------------------------------------------------- */

template <typename T> __global__ void
im2row_backward_kernel(T* data,
                        T const* stacked,
                       const int numPatchesX,
                       const int numPatchesY,
                       const int dataVolume,
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
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < dataVolume)
  {
    T accumulator = 0 ;
    /*
     This kernel accumulates on data[index] all elements in stacked
     that receive copies of data[index] in im2row.

     Consider coordinate (x_data,y_data) in the input image. Relative to patch
     (x,y), this has offset

     u = x_data - (x * strideX - padLeft)
     v = y_data - (y * strideY - padRight)

     In particular, (x_data,y_data) is contained (and hence contributes)
     to patch (x,y) if, and only if,

     0 <= u < windowWidth  <==>  1) x_data >= x * strideX - padLeft
     2) x_data <  x * strideX - padLeft + windowWidth

     and similar for y.

     Hence, the patches that contribute to (x_data,y_data) are given
     by indexes (x,y) such that

     (x_data + padLeft - windowWidth)/stride < x
     <= (x_data + padLeft)/stride

     or, accounting for the boundaries,

     x1 <= x <= x2, such that
     x1 = max(0,  1 + floor(x_data + padLeft - windowWidth)/stride),
     x2 = min(numPatchesX-1,  floor(x_data + padLeft)/stride),

     and similar for y.

     Note that (x_data + padLeft - windowWidth) may be negative. In this case,
     the C convention for rounding division towards zero fails to compute
     the floor() properly. Instead, we check this case explicitly and set
     */

    int x_data = index ;
    int y_data = x_data / width ;
    int z = y_data / height ;
    x_data %= width ;
    y_data %= height ;

    int dx = x_data + padLeft - windowWidth ;
    int dy = y_data + padTop - windowHeight ;
    int x1 = (dx >= 0) ? dx/strideX + 1 : 0 ;
    int y1 = (dy >= 0) ? dy/strideY + 1 : 0 ;
    int x2 = min((x_data + padLeft) / strideX, numPatchesX - 1) ;
    int y2 = min((y_data + padTop) / strideY, numPatchesY - 1) ;

    /*
     Knowing which patches (x,y) contribute to (x_data,y_data) is not enough;
     we need to determine the specific element within each patch. This
     is given by the offset as given above:

     u(x) = x_data - (x * strideX - padLeft)
     v(y) = y_data - (y * strideY - padRight)

     Now we can comptute the indeces of the elements of stacked[] to accumulate:

     stackedIndex(x,y) =
     (y * numPatchesX + x) +                 // column offset
     ((z * windowHeight + v(y)) * windowWidth + u(x)) *  // within patch offset
     (numPatchesX*numPatchesY)

     Substituting the expression fo u(x), we find

     stackedIndex(x,y) =
     = (y * numPatchesX + x)
     + ((z * windowHeight + y_data + padTop) * windowWidth + x_data + padLeft)
     * (numPatchesX*numPatchesY)
     - ((y * strideY) * windowWidth + x * strideX)
     * (numPatchesX*numPatchesY)
     = (z * windowHeight + y_data + padTop) * windowWidth + x_data + padLeft)
     + x * (1 - strideX*numPatchesY*numPatchesX)
     + y * (1 - strideY*numPatchesY*windowWidth)*numPatchesX ;

     */

    int deltax = (1 - strideX * numPatchesY * numPatchesX) ;
    int deltay = (1 - strideY * numPatchesY * windowWidth) * numPatchesX ;
    stacked += ((z * windowHeight + y_data + padTop) * windowWidth + (x_data + padLeft)) * (numPatchesX*numPatchesY) ;

    for (int y = y1 ; y <= y2 ; ++ y) {
      for (int x = x1 ; x <= x2 ; ++ x) {
        accumulator += stacked[y * deltay + x * deltax];
      }
    }
    data[index] = accumulator;
  }
}

/* ---------------------------------------------------------------- */
/*                             im2row backward kernel with dilation */
/* ---------------------------------------------------------------- */

template <typename T> __global__ void
im2row_backward_kernel(T* data,
                        T const* stacked,
                       const int numPatchesX,
                       const int numPatchesY,
                       const int dataVolume,
                       const int width,
                       const int height,
                       const int depth,
                       const int windowWidth,
                       const int windowHeight,
                       const int strideX,
                       const int strideY,
                       const int padLeft,
                       const int padTop,
                       const int dilateX,
                       const int dilateY)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int nWindowWidth = ((windowWidth - 1) * dilateX + 1);
  int nWindowHeight = ((windowHeight - 1) * dilateY + 1);

  if (index < dataVolume)
  {
    T accumulator = 0 ;

    int x_data = index ;
    int y_data = x_data / width ;
    int z = y_data / height ;
    x_data %= width ;
    y_data %= height ;

    int dx = x_data + padLeft - nWindowWidth ;
    int dy = y_data + padTop - nWindowHeight ;
    int x1 = (dx >= 0) ? dx/strideX + 1 : 0 ;
    int y1 = (dy >= 0) ? dy/strideY + 1 : 0 ;
    int x2 = min((x_data + padLeft) / strideX, numPatchesX - 1) ;
    int y2 = min((y_data + padTop) / strideY, numPatchesY - 1) ;

    stacked += ((z * windowHeight) * windowWidth) * (numPatchesX*numPatchesY);
    for (int y = y1 ; y <= y2 ; ++y ) {
      int v_y = (y_data - (y * strideY - padTop));
      if (v_y % dilateY != 0){
        continue;
      }
      v_y = v_y/dilateY;
      for (int x = x1 ; x <= x2 ; ++x ) {
        int u_x = (x_data - (x * strideX - padLeft));
        if (u_x % dilateX != 0){
          continue;
        }
        u_x = u_x/dilateX;
        int ptr = (y * numPatchesX + x) + (u_x + v_y * windowWidth) * (numPatchesX*numPatchesY);
        accumulator += stacked[ptr];
      }
    }
    data[index] = accumulator;
  }
}

namespace vl { namespace impl {

  template<typename type>
  struct im2row<vl::GPU, type>
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
      /* Each kernel instance copies a feature dimension of a patch */
      int nWindowWidth = ((windowWidth - 1) * dilateX + 1);
      int nWindowHeight = ((windowHeight - 1) * dilateY + 1);
      int numPatchesX = (width + (padLeft + padRight) - nWindowWidth)/strideX + 1 ;
      int numPatchesY = (height + (padTop + padBottom) - nWindowHeight)/strideY + 1 ;
      int numPatchSlices = numPatchesX * numPatchesY * depth ;

      im2row_forward_kernel<type>
      <<< divideUpwards(numPatchSlices, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
      (stacked,
       data,
       numPatchesX,
       numPatchesY,
       numPatchSlices,
       width, height,
       windowWidth, windowHeight,
       strideX, strideY,
       padLeft, padTop,
       dilateX, dilateY) ;

      return context.setError(context.getCudaHelper().catchCudaError(__func__)) ;
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
      /*
       Each kernel integrates all contributions to a particular element
       of data.
       */
      int nWindowWidth = ((windowWidth - 1) * dilateX + 1);
      int nWindowHeight = ((windowHeight - 1) * dilateY + 1);
      int numPatchesX = (width + (padLeft + padRight) - nWindowWidth)/strideX + 1 ;
      int numPatchesY = (height + (padTop + padBottom) - nWindowHeight)/strideY + 1 ;

      int dataVolume = width * height * depth ;

      /*
        The backward operation for dilated convolution is not as efficient
        as the backward operation for regular convolutions, so if no dilation
        is necessary, we use the previous implementation.
      */
      if (dilateX == 1 && dilateY == 1){
        im2row_backward_kernel<type>
        <<< divideUpwards(dataVolume, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
        (data,
         stacked,
         numPatchesX,
         numPatchesY,
         dataVolume,
         width, height, depth,
         windowWidth, windowHeight,
         strideX, strideY,
         padLeft, padTop) ;
      }else{
        im2row_backward_kernel<type>
        <<< divideUpwards(dataVolume, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
        (data,
         stacked,
         numPatchesX,
         numPatchesY,
         dataVolume,
         width, height, depth,
         windowWidth, windowHeight,
         strideX, strideY,
         padLeft, padTop,
         dilateX, dilateY) ;
      }
      return context.setError(context.getCudaHelper().catchCudaError(__func__)) ;
    }

  } ;

} }

// Instantiations
template struct vl::impl::im2row<vl::GPU, float> ;

#ifdef ENABLE_DOUBLE
template struct vl::impl::im2row<vl::GPU, double> ;
#endif
