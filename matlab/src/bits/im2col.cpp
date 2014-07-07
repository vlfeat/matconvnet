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

template <typename Dtype>
void im2col_cpu(const Dtype* data_im,
                const int channels, const int height, const int width,
                const int kheight, const int kwidth,
                const int stride, const int pad,
                Dtype* data_col)
{
  // the input is an image array of size H,W,C
  // this functin prepares an array for filtering with filres of size ksize^2
  // this function creates a new array of size H/stride, W/stride C*ksize^2 (crazy large!)
  int height_col = (height + 2 * pad - kheight) / stride + 1;
  int width_col = (width + 2 * pad - kwidth) / stride + 1;
  int channels_col = channels * kheight * kwidth;
  for (int c = 0; c < channels_col; ++c) {
    int w_offset = c % kwidth;
    int h_offset = (c / kwidth) % kheight;
    int c_im = c / kheight / kwidth;

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
    int y0 =  static_max(0, ceil_divide(pad - h_offset, stride)) ;
    int y1 =  static_min(height_col, ceil_divide(height + pad - h_offset, stride)) ;

    for (int y = 0 ; y < y0 ; ++y) {
      for (int x = 0 ; x < width_col ; ++x) {
        data_col[(c * height_col + y) * width_col + x] = 0 ;
      }
    }
    for (int y = y0 ; y < y1 ; ++y) {
      int x0 =  static_max(0, ceil_divide(pad - w_offset, stride)) ;
      int x1 =  static_min(width_col,  ceil_divide(width  + pad - w_offset, stride)) ;
      const int y_im = y * stride + h_offset - pad;
      const int x_im = x0 * stride + w_offset - pad;
      Dtype * a = data_col + (c * height_col + y) * width_col + x0 ;
      Dtype const * b = data_im + (c_im * height + y_im) * width + x_im ;

      for (int x = 0 ; x < x0 ; ++x) {
        data_col[(c * height_col + y) * width_col + x] = 0 ;
      }
      for (int x = x0 ; x < x1 ; ++x) {
        *a = *b ;
        a += 1 ;
        b += stride ;
      }
      for (int x = x1 ; x < width_col ; ++x) {
        data_col[(c * height_col + y) * width_col + x] = 0 ;
      }
    }
    for (int y = y1 ; y < height_col ; ++y) {
      for (int x = 0 ; x < width_col ; ++x) {
        data_col[(c * height_col + y) * width_col + x] = 0 ;
      }
    }
#endif
  }
}

template void im2col_cpu<float>(const float* data_im, const int channels,
                                const int height, const int width, const int kheight, const int kwidth,
                                const int pad, const int stride,
                                float* data_col);

template void im2col_cpu<double>(const double* data_im, const int channels,
                                 const int height, const int width, const int kheight, const int kwidth,
                                 const int pad, const int stride,
                                 double* data_col);

#if 0
/* ---------------------------------------------------------------- */
/*                                                     im2row (CPU) */
/* ---------------------------------------------------------------- */

template <typename Dtype>
void im2row_cpu(const Dtype* data_im,
                const int channels, const int height, const int width,
                const int kheight, const int kwidth, const int stride,
                Dtype* data_col)
{
  int height_col = (height - kheight) / stride + 1;
  int width_col = (width - kwidth) / stride + 1;
  int channels_col = channels * kheight * kwidth;
  for (int y = 0; y < height_col; ++y)
    for (int x = 0; x < width_col; ++x)
      Dtype * patch_out = &data_col[(h * width_col + w) * channels_col] ;
      Dtype const * patch_in = &data_im[h * width + w] ;
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

template <typename Dtype>
void col2im_cpu(const Dtype* data_col, const int channels,
                const int height, const int width, const int kheight, const int kwidth,
                const int stride, const int pad,
                Dtype* data_im) {
  memset(data_im, 0, sizeof(Dtype) * width * height * channels);
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


