//
//  im2col.cpp
//  matconv
//
//  Created by Andrea Vedaldi on 09/02/2014.
//  Copyright (c) 2014 Andrea Vedaldi. All rights reserved.
//

#include "im2col.hpp"

// Hack caffe away

#define FATAL std::cout
#define LOG(x) x

#define CUDA_POST_KERNEL_CHECK \
if (cudaSuccess != cudaPeekAtLastError()) \
LOG(FATAL) << "[Caffe]: Cuda kernel failed. Error: " \
<< cudaGetErrorString(cudaPeekAtLastError()) << std::endl

// We will use 1024 threads per block, which requires cuda sm_2x or above.
#if __CUDA_ARCH__ >= 200
const int CAFFE_CUDA_NUM_THREADS = 1024;
#else
const int CAFFE_CUDA_NUM_THREADS = 512;
#endif

inline int CAFFE_GET_BLOCKS(const int N) {
  return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
}


/* ---------------------------------------------------------------- */
/*                                                     im2col (CPU) */
/* ---------------------------------------------------------------- */

template <typename Dtype>
void im2col_cpu(const Dtype* data_im,
                const int channels, const int height, const int width,
                const int ksize, const int stride,
                Dtype* data_col)
{
  // the input is an image array of size H,W,C
  // this functin prepares an array for filtering with filres of size ksize^2
  // this function creates a new array of size H/stride, W/stride C*ksize^2 (crazy large!)
  int height_col = (height - ksize) / stride + 1;
  int width_col = (width - ksize) / stride + 1;
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
    for (int h = 0; h < height_col; ++h) {
      for (int w = 0; w < width_col; ++w) {
        data_col[(c * height_col + h) * width_col + w] =
        data_im[(c_im * height + h * stride + h_offset) * width
                + w * stride + w_offset];
      }
    }
  }
}

template void im2col_cpu<float>(const float* data_im, const int channels,
                                const int height, const int width, const int ksize, const int stride,
                                float* data_col);

template void im2col_cpu<double>(const double* data_im, const int channels,
                                 const int height, const int width, const int ksize, const int stride,
                                 double* data_col);

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
  for (int h = 0; h < height_col; ++h) {
    for (int w = 0; w < width_col; ++w) {
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

/* ---------------------------------------------------------------- */
/*                                                     col2im (CPU) */
/* ---------------------------------------------------------------- */

template <typename Dtype>
void col2im_cpu(const Dtype* data_col, const int channels,
                const int height, const int width, const int ksize, const int stride,
                Dtype* data_im) {
  memset(data_im, 0, sizeof(Dtype) * height * width * channels);
  int height_col = (height - ksize) / stride + 1;
  int width_col = (width - ksize) / stride + 1;
  int channels_col = channels * ksize * ksize;
  for (int c = 0; c < channels_col; ++c) {
    int w_offset = c % ksize;
    int h_offset = (c / ksize) % ksize;
    int c_im = c / ksize / ksize;
    for (int h = 0; h < height_col; ++h) {
      for (int w = 0; w < width_col; ++w) {
        data_im[(c_im * height + h * stride + h_offset) * width + w * stride
                + w_offset] += data_col[(c * height_col + h) * width_col + w];
      }
    }
  }
}

template void col2im_cpu<float>(const float* data_col, const int channels,
                                const int height, const int width, const int psize, const int stride,
                                float* data_im);

template void col2im_cpu<double>(const double* data_col, const int channels,
                                 const int height, const int width, const int psize, const int stride,
                                 double* data_im);

/* ---------------------------------------------------------------- */
/*                                                     im2col (GPU) */
/* ---------------------------------------------------------------- */

template <typename Dtype>
__global__ void im2col_gpu_kernel(const int n, const Dtype* data_im,
                                  const int height, const int width, const int ksize,
                                  const int stride, const int height_col, const int width_col, Dtype* data_col) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < n) {
    int w_out = index % width_col;
    index /= width_col;
    int h_out = index % height_col;
    int channel_in = index / height_col;
    int channel_out = channel_in * ksize * ksize;
    int h_in = h_out * stride;
    int w_in = w_out * stride;
    data_col += (channel_out * height_col + h_out) * width_col + w_out;
    data_im += (channel_in * height + h_in) * width + w_in;
    for (int i = 0; i < ksize; ++i) {
      for (int j = 0; j < ksize; ++j) {
        *data_col = data_im[i * width + j];
        data_col += height_col * width_col;
      }
    }
  }
}

template <typename Dtype>
void im2col_gpu(const Dtype* data_im, const int channels,
                const int height, const int width, const int ksize, const int stride,
                Dtype* data_col) {
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int height_col = (height - ksize) / stride + 1;
  int width_col = (width - ksize) / stride + 1;
  int num_kernels = channels * height_col * width_col;
  im2col_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>
  (num_kernels, data_im, height, width, ksize, stride, height_col, width_col, data_col);
  CUDA_POST_KERNEL_CHECK;
}

// Explicit instantiation
template void im2col_gpu<float>(const float* data_im, const int channels,
                                const int height, const int width, const int ksize, const int stride,
                                float* data_col);

template void im2col_gpu<double>(const double* data_im, const int channels,
                                 const int height, const int width, const int ksize, const int stride,
                                 double* data_col);

/* ---------------------------------------------------------------- */
/*                                                     col2im (GPU) */
/* ---------------------------------------------------------------- */

template <typename Dtype>
__global__ void col2im_gpu_kernel(const int n, const Dtype* data_col,
                                  const int height, const int width, const int channels, const int ksize,
                                  const int stride, const int height_col, const int width_col, Dtype* data_im) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < n) {
    Dtype val = 0;
    int w = index % width;
    int h = (index / width) % height;
    int c = index / (width * height);
    // compute the start and end of the output
    int w_col_start = (w < ksize) ? 0 : (w - ksize) / stride + 1;
    int w_col_end = min(w / stride + 1, width_col);
    int h_col_start = (h < ksize) ? 0 : (h - ksize) / stride + 1;
    int h_col_end = min(h / stride + 1, height_col);
    /*
     for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
     for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
     // the col location: [c * width * height + h_out, w_out]
     int c_col = c * ksize * ksize + (h - h_col * stride) * ksize + (w - w_col * stride);
     val += data_col[(c_col * height_col + h_col) * width_col + w_col];
     }
     }
     */
    // equivalent implementation
    int offset = (c * ksize * ksize + h * ksize + w) * height_col * width_col;
    int coeff_h_col = (1 - stride * ksize * height_col) * width_col;
    int coeff_w_col = (1 - stride * height_col * width_col);
    for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
      for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
        val += data_col[offset + h_col * coeff_h_col + w_col * coeff_w_col];
      }
    }
    data_im[index] = val;
  }
}

template <typename Dtype>
void col2im_gpu(const Dtype* data_col, const int channels,
                const int height, const int width, const int ksize, const int stride,
                Dtype* data_im) {
  //CUDA_CHECK(cudaMemset(data_im, 0, sizeof(Dtype) * height * width * channels));
  int height_col = (height - ksize) / stride + 1;
  int width_col = (width - ksize) / stride + 1;
  int num_kernels = channels * height * width;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  col2im_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(num_kernels, data_col, height, width, channels, ksize, stride, height_col, width_col, data_im) ;
  CUDA_POST_KERNEL_CHECK;
}

template void col2im_gpu<float>(const float* data_col, const int channels,
                                const int height, const int width, const int psize, const int stride,
                                float* data_im);

template void col2im_gpu<double>(const double* data_col, const int channels,
                                 const int height, const int width, const int psize, const int stride,
                                 double* data_im);
