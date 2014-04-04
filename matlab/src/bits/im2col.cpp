/** @file im2col.cpp
 ** @brief Image to columns and back (CPU)
 ** @author Andrea Vedaldi
 **/

#include "im2col.hpp"
#include <string.h>

/* ---------------------------------------------------------------- */
/*                                                     im2col (CPU) */
/* ---------------------------------------------------------------- */

template <typename Dtype>
void im2col_cpu(const Dtype* data_im,
                const int channels, const int height, const int width,
                const int ksize, const int stride, const int pad,
                Dtype* data_col)
{
  // the input is an image array of size H,W,C
  // this functin prepares an array for filtering with filres of size ksize^2
  // this function creates a new array of size H/stride, W/stride C*ksize^2 (crazy large!)
  int height_col = (height + 2 * pad - ksize) / stride + 1;
  int width_col = (width + 2 * pad - ksize) / stride + 1;
  int channels_col = channels * ksize * ksize;
  for (int c = 0; c < channels_col; ++c) {
    int w_offset = c % ksize;
    int h_offset = (c / ksize) % ksize;
    int c_im = c / ksize / ksize;
    for (int y = 0; y < height_col; ++y) {
      for (int x = 0; x < width_col; ++x) {
        const int y_im = y * stride + h_offset - pad;
        const int x_im = x * stride + w_offset - pad;
        // check if we are copying from the padded (outside the image) area
        if (y_im >=0 && y_im < height && x_im >=0 && x_im < width) {
          data_col[(c * height_col + y) * width_col + x] =
            data_im[(c_im * height + y_im) * width + x_im] ;
        } else {
          data_col[(c * height_col + y) * width_col + x] = 0 ;
        }
      }
    }
  }
}

template void im2col_cpu<float>(const float* data_im, const int channels,
                                const int height, const int width, const int ksize,
                                const int pad, const int stride,
                                float* data_col);

template void im2col_cpu<double>(const double* data_im, const int channels,
                                 const int height, const int width, const int ksize,
                                 const int pad, const int stride,
                                 double* data_col);

#if 0
/* ---------------------------------------------------------------- */
/*                                                     im2row (CPU) */
/* ---------------------------------------------------------------- */

template <typename Dtype>
void im2row_cpu(const Dtype* data_im,
                const int channels, const int height, const int width,
                const int ksize, const int stride,
                Dtype* data_col)
{
  int height_col = (height - ksize) / stride + 1;
  int width_col = (width - ksize) / stride + 1;
  int channels_col = channels * ksize * ksize;
  for (int y = 0; y < height_col; ++y)
    for (int x = 0; x < width_col; ++x)
      Dtype * patch_out = &data_col[(h * width_col + w) * channels_col] ;
      Dtype const * patch_in = &data_im[h * width + w] ;
      for (int c = 0; c < channels ; ++c) {
        for (int hh = 0 ; hh < ksize ; ++hh) {
          for(int ww = 0 ; ww < ksize ; ++ww) {
            patch_out[hh*ksize + ww] = patch_in[hh*width + ww] ;
          }
        }
        patch_out += ksize*ksize ;
        patch_in += width*height ;
      }
    }
  }
}

template void im2row_cpu<float>(const float* data_im, const int channels,
                                const int height, const int width, const int ksize, const int stride,
                                float* data_col);

template void im2row_cpu<double>(const double* data_im, const int channels,
                                 const int height, const int width, const int ksize, const int stride,
                                 double* data_col);
#endif

/* ---------------------------------------------------------------- */
/*                                                     col2im (CPU) */
/* ---------------------------------------------------------------- */

template <typename Dtype>
void col2im_cpu(const Dtype* data_col, const int channels,
                const int height, const int width, const int ksize,
                const int stride, const int pad,
                Dtype* data_im) {
  memset(data_im, 0, sizeof(Dtype) * width * height * channels);
  int height_col = (height + 2 * pad - ksize) / stride + 1;
  int width_col = (width + 2 * pad - ksize) / stride + 1;
  int channels_col = channels * ksize * ksize;
  for (int c = 0; c < channels_col; ++c) {
    /* in the filter volume, filter element c has this spatial/channel offset */
    int w_offset = c % ksize;
    int h_offset = (c / ksize) % ksize;
    int c_im = c / ksize / ksize;
    /* now scan all the filter applications */
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
  }
}

template void col2im_cpu<float>(const float* data_col, const int channels,
                                const int height, const int width, const int ksize,
                                const int stride, const int pad,
                                float* data_im);

template void col2im_cpu<double>(const double* data_col, const int channels,
                                 const int height, const int width, const int ksize,
                                 const int stride, const int pad,
                                 double* data_im);


