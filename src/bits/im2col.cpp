//
//  im2col.cpp
//  matconv
//
//  Created by Andrea Vedaldi on 09/02/2014.
//  Copyright (c) 2014 Andrea Vedaldi. All rights reserved.
//

#include "caffe-scraps.hpp"
#include "im2col.hpp"

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
  // the output has C*ksize^2 layers; fill one layer per time
  // the first output channel copies the first input channels with zero shift
  // the second outout chanel still copies the first input channels, but shifted by one pixes
  // this layout is designed such that filtering reduces directly to a matrix multiplication with the
  // filer arrays
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

/* ---------------------------------------------------------------- */
/*                                                     im2col (GPU) */
/* ---------------------------------------------------------------- */

template <typename Dtype>
__global__ void
im2col_gpu_kernel(const int n, const Dtype* data_im,
                  const int height, const int width, const int ksize,
                  const int stride, const int pad,
                  const int height_col, const int width_col, Dtype* data_col)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < n) {
    int x_out = index % width_col;
    index /= width_col;
    int y_out = index % height_col;
    int channel_in = index / height_col;
    int channel_out = channel_in * ksize * ksize;
    int y_in = y_out * stride - pad;
    int x_in = x_out * stride - pad;
    data_col += (channel_out * height_col + y_out) * width_col + x_out;
    data_im += (channel_in * height + y_in) * width + x_in;
    for (int i = 0; i < ksize; ++i) {
      for (int j = 0; j < ksize; ++j) {
        if (y_in + i >= 0 && y_in + i < height && x_in + j >= 0 && x_in + j < width) {
          *data_col = data_im[i * width + j];
        } else {
          *data_col = 0;
        }
        data_col += height_col * width_col;
      }
    }
  }
}

template <typename Dtype>
void im2col_gpu(const Dtype* data_im, const int channels,
                const int height, const int width, const int ksize,
                const int pad, const int stride,
                Dtype* data_col)
{
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int height_col = (height + 2 * pad - ksize) / stride + 1;
  int width_col = (width + 2 * pad - ksize) / stride + 1;
  int num_kernels = channels * height_col * width_col;
  im2col_gpu_kernel<Dtype> <<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>
  (num_kernels, data_im, height, width, ksize, stride, pad, height_col, width_col, data_col);
  CUDA_POST_KERNEL_CHECK;
}

// Explicit instantiation
template void im2col_gpu<float>(const float* data_im, const int channels,
                                const int height, const int width, const int ksize,
                                const int stride, const int pad,
                                float* data_col);

template void im2col_gpu<double>(const double* data_im, const int channels,
                                 const int height, const int width, const int ksize,
                                 const int stride, const int pad,
                                 double* data_col);

/* ---------------------------------------------------------------- */
/*                                                     col2im (GPU) */
/* ---------------------------------------------------------------- */

template <typename Dtype>
__global__ void col2im_gpu_kernel(const int n, const Dtype* data_col,
                                  const int height, const int width,
                                  const int channels, const int ksize,
                                  const int pad, const int stride,
                                  const int height_col, const int width_col,
                                  Dtype* data_im)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;

  if (index < n)
  {
    Dtype val = 0;
    // derive kernel's (x_in, y_in, channel_in) parameters, and take padding into account
    int x = (index % width) + pad;
    int y = ((index / width) % height) + pad;
    int c = index / (width * height);
    // compute the start and end of the output
    int x_col_start = (x < ksize) ? 0 : (x - ksize) / stride + 1;
    int x_col_end = min(x / stride + 1, width_col);
    int y_col_start = (y < ksize) ? 0 : (y - ksize) / stride + 1;
    int y_col_end = min(y / stride + 1, height_col);
    //        for (int y_col = y_col_start; y_col < y_col_end; ++y_col)
    //        {
    //            for (int x_col = x_col_start; x_col < x_col_end; ++x_col)
    //            {
    //                // find the output channel which corresponds to the input channel and (x_col, y_col)
    //                int c_col = c * ksize * ksize + (y - y_col * stride) * ksize + (x - x_col * stride);
    //                val += data_col[(c_col * height_col + y_col) * width_col + x_col];
    //            }
    //        }

    // equivalent implementation, with some operations taken out of the loop body
    int offset = (c * ksize * ksize + y * ksize + x) * height_col * width_col;
    int coeff_y_col = (1 - stride * ksize * height_col) * width_col;
    int coeff_x_col = (1 - stride * height_col * width_col);
    for (int y_col = y_col_start; y_col < y_col_end; ++y_col) {
      for (int x_col = x_col_start; x_col < x_col_end; ++x_col) {
        val += data_col[offset + y_col * coeff_y_col + x_col * coeff_x_col];
      }
    }
    data_im[index] = val;
  }
}

template <typename Dtype>
void col2im_gpu(const Dtype* data_col, const int channels,
                const int height, const int width, const int ksize,
                const int pad, const int stride,
                Dtype* data_im)
{
  int height_col = (height + 2 * pad - ksize) / stride + 1;
  int width_col = (width + 2 * pad - ksize) / stride + 1;
  int num_kernels = channels * height * width;
  col2im_gpu_kernel<Dtype> <<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>
  (num_kernels, data_col, height, width, channels, ksize, stride, pad,
   height_col, width_col, data_im);
  CUDA_POST_KERNEL_CHECK;
}

template void col2im_gpu<float>(const float* data_col, const int channels,
                                const int height, const int width, const int ksize,
                                const int stride, const int pad,
                                float* data_im);

template void col2im_gpu<double>(const double* data_col, const int channels,
                                 const int height, const int width, const int ksize,
                                 const int stride, const int pad,
                                 double* data_im);
