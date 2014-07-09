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
  int numCols = numPatchesX * numPatchesY ;
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

#if 0
    for (int y = 0; y < height_col; ++y) {
      for (int x = 0; x < width_col; ++x) {
        const int y_im = y * stride + h_offset - pad;
        const int x_im = x * stride + w_offset - pad;
        // check if we are copying from the padded (outside the image) area
        if (y_im >=0 &&
            y_im < height &&
            x_im >=0 &&
            x_im < width) {
          data_col[(c * height_col + y) * width_col + x] =
            data_im[(c_im * height + y_im) * width + x_im] ;
        } else {
          data_col[(c * height_col + y) * width_col + x] = 0 ;
        }
      }
    }
#else
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

    int y0 = static_max(0, ceil_divide(padTop - v, strideY)) ;
    int y1 = static_min(numPatchesY, ceil_divide(height + padTop - v, strideY)) ;
    for (int y = 0 ; y < y0 ; ++y) {
      for (int x = 0 ; x < numPatchesX ; ++x) {
        *stacked++ = 0 ;
      }
    }

    for (int y = y0 ; y < y1 ; ++y) {
      int x0 = static_max(0, ceil_divide(padLeft - u, strideX)) ;
      int x1 = static_min(numPatchesX,  ceil_divide(width + padLeft - u, strideX)) ;
      const int y_data = y * strideY + v - padTop ;
      const int x_data = x0 * strideX + u - padLeft ;
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
#endif
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

#if 0
/* ---------------------------------------------------------------- */
/*                                                     im2row (CPU) */
/* ---------------------------------------------------------------- */

template <typename T>
void im2row_cpu(const T* data_im,
                const int channels, const int height, const int width,
                const int kheight, const int kwidth, const int stride,
                T* data_col)
{
  int height_col = (height - kheight) / stride + 1;
  int width_col = (width - kwidth) / stride + 1;
  int channels_col = channels * kheight * kwidth;
  for (int y = 0; y < height_col; ++y)
    for (int x = 0; x < width_col; ++x)
      T * patch_out = &data_col[(h * width_col + w) * channels_col] ;
      T const * patch_in = &data_im[h * width + w] ;
      for (int c = 0; c < channels ; ++c) {
        for (int hh = 0 ; hh < kheight ; ++hh) {
          for(int ww = 0 ; ww < kwidth ; ++ww) {
            patch_out[hh*kwidth + ww] = patch_in[hh*width + ww] ;
          }
        }
        patch_out += kwidth*kheight ;
        patch_in += width*height ;
      }
    }
  }
}

template void im2row_cpu<float>(const float* data_im, const int channels,
                                const int height, const int width, const int kheight, const int kwidth, const int stride,
                                float* data_col);

template void im2row_cpu<double>(const double* data_im, const int channels,
                                 const int height, const int width, const int kheight, const int kwidth, const int stride,
                                 double* data_col);
#endif

/* ---------------------------------------------------------------- */
/*                                                     col2im (CPU) */
/* ---------------------------------------------------------------- */

template <typename T>
void col2im_cpu(const T* data_col, const int channels,
                const int height, const int width, const int kheight, const int kwidth,
                const int stride, const int pad,
                T* data_im) {
  memset(data_im, 0, sizeof(T) * width * height * channels);
  int height_col = (height + 2 * pad - kheight) / stride + 1;
  int width_col = (width + 2 * pad - kwidth) / stride + 1;
  int channels_col = channels * kheight * kwidth;
  for (int c = 0; c < channels_col; ++c) {
    /* in the filter volume, filter element c has this spatial/channel offset */
    int w_offset = c % kwidth;
    int h_offset = (c / kwidth) % kheight;
    int c_im = c / kwidth / kheight;
    /* now scan all the filter applications */

#if 0
    for (int y = 0; y < height_col; ++y) {
      for (int x = 0; x < width_col; ++x) {
        const int y_im = y * stride + h_offset - pad ;
        const int x_im = x * stride + w_offset - pad ;
        if (y_im >= 0 && y_im < height && x_im >= 0 && x_im < width) {
          data_im[(c_im * height + y_im) * width + x_im] +=
          data_col[(c * height_col + y) * width_col + x];
        }
      }
    }
#else
    /* y stride + h_offset - p >= 0
       y >= max(0, (p - h_offset)/stride) [should pick ceil]

       all quantities are integer
       y stride + h_offset - pad < height
       y < (height + pad - h_offset)/stride
       y <= ceil_divide(height + pad - h_offset,stride)-1

       y < min(height_col, (height + pad - h_offset)/stride)
    */
    int y0 =  static_max(0, ceil_divide(pad - h_offset, stride)) ;
    int x0 =  static_max(0, ceil_divide(pad - w_offset, stride)) ;
    int y1 =  static_min(height_col, ceil_divide(height + pad - h_offset, stride)) ;
    int x1 =  static_min(width_col,  ceil_divide(width  + pad - w_offset, stride)) ;

    for (int y = y0 ; y < y1 ; ++y) {
      for (int x = x0; x < x1 ; ++x) {
        const int y_im = y * stride + h_offset - pad ;
        const int x_im = x * stride + w_offset - pad ;
        data_im[(c_im * height + y_im) * width + x_im] +=
          data_col[(c * height_col + y) * width_col + x];
      }
    }
#endif
  }
}

template void col2im_cpu<float>(const float* data_col, const int channels,
                                const int height, const int width, const int kheight, const int kwidth,
                                const int stride, const int pad,
                                float* data_im);

template void col2im_cpu<double>(const double* data_col, const int channels,
                                 const int height, const int width, const int kheight, const int kwidth,
                                 const int stride, const int pad,
                                 double* data_im);


