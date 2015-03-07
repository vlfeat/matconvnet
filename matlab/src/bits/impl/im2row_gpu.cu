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
im2row_gpu_kernel(T* stacked,
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
                  const int padTop)
{
  /* each kernel copies the pixels in an image patch for one channel */
  int index = threadIdx.x + blockIdx.x * blockDim.x ;
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
    for (int v = 0 ; v < windowHeight ; ++v) {
      for (int u = 0 ; u < windowWidth ; ++u) {
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

template <typename T> static inline cudaError_t
im2row_gpu(T* stacked,
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
  /* Each kernel instance copies a feature dimension of a patch */

  int numPatchesX = (width + (padLeft + padRight) - windowWidth)/strideX + 1 ;
  int numPatchesY = (height + (padTop + padBottom) - windowHeight)/strideY + 1 ;
  int numPatchSlices = numPatchesX * numPatchesY * depth ;

  im2row_gpu_kernel<T>
  <<< divideUpwards(numPatchSlices, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
  (stacked,
   data,
   numPatchesX,
   numPatchesY,
   numPatchSlices,
   width, height,
   windowWidth, windowHeight,
   strideX, strideY,
   padLeft, padTop) ;

  return cudaPeekAtLastError() ;
}


template <> vl::Error
vl::impl::im2row<vl::GPU, float>(vl::Context& context,
                                 float* stacked,
                                 float const* data,
                                 size_t height, size_t width, size_t depth,
                                 size_t windowHeight, size_t windowWidth,
                                 size_t strideY, size_t strideX,
                                 size_t padTop, size_t padBottom, size_t padLeft, size_t padRight)
{
  int status ;
  status = im2row_gpu<float>(stacked, data,
                             height, width, depth,
                             windowHeight, windowWidth,
                             strideY, strideX,
                             padTop, padBottom, padLeft, padRight) ;
  return (status == cudaSuccess) ? vl::vlSuccess : vl::vlErrorCuda ;
}


/* ---------------------------------------------------------------- */
/*                                                           row2im */
/* ---------------------------------------------------------------- */

template <typename T>
__global__ void row2im_gpu_kernel(T* data,
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

template <typename T> static inline cudaError_t
row2im_gpu(T* data,
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
  /*
   Each kernel integrates all contributions to a particular element
   of data.
   */

  int numPatchesX = (width + (padLeft + padRight) - windowWidth)/strideX + 1 ;
  int numPatchesY = (height + (padTop + padBottom) - windowHeight)/strideY + 1 ;
  int dataVolume = width * height * depth ;

  row2im_gpu_kernel<T>
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

  return cudaPeekAtLastError() ;
}

template <> vl::Error
vl::impl::row2im<vl::GPU, float>(vl::Context& context,
                                 float* data,
                                 float const* stacked,
                                 size_t height, size_t width, size_t depth,
                                 size_t windowHeight, size_t windowWidth,
                                 size_t strideY, size_t strideX,
                                 size_t padTop, size_t padBottom, size_t padLeft, size_t padRight)
{
  int status ;
  status = row2im_gpu<float>(data, stacked,
                             height, width, depth,
                             windowHeight, windowWidth,
                             strideY, strideX,
                             padTop, padBottom, padLeft, padRight) ;
  return (status == cudaSuccess) ? vl::vlSuccess : vl::vlErrorCuda ;
}
