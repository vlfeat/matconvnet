//
//  pooling.cpp
//  matconv
//
//  Created by Andrea Vedaldi on 11/02/2014.
//  Copyright (c) 2014 Andrea Vedaldi. All rights reserved.
//

#include "caffe-scraps.hpp"
#include "pooling.hpp" 

/* ---------------------------------------------------------------- */
/*                                                 maxPooling (CPU) */
/* ---------------------------------------------------------------- */

template<typename T>
void maxPooling_cpu(T* pooled,
                    T const* data,
                    size_t width,
                    size_t height,
                    size_t depth,
                    size_t poolSize,
                    size_t poolStride)
{
  int pooledWidth = (width - poolSize)/poolStride + 1 ;
  int pooledHeight = (height - poolSize)/poolStride + 1 ;

  for (int c = 0; c < depth; ++c) {
    for (int ph = 0; ph < pooledHeight; ++ph) {
      for (int pw = 0; pw < pooledWidth; ++pw) {
        int hstart = ph * poolStride;
        int wstart = pw * poolStride;
        int hend = std::min(hstart + poolSize, height);
        int wend = std::min(wstart + poolSize, width);
        for (int h = hstart; h < hend; ++h) {
          for (int w = wstart; w < wend; ++w) {
            pooled[ph * pooledWidth + pw] = max(pooled[ph * pooledWidth + pw],
                                                data[h * width + w]);
          }
        }
      }
    }
    data += width*height ;
    pooled += pooledWidth*pooledHeight ;
  }
}

template
void maxPooling_cpu<float>(float* pooled,
                           float const* data,
                           size_t width,
                           size_t height,
                           size_t depth,
                           size_t poolSize,
                           size_t poolStride) ;

template
void maxPooling_cpu<double>(double* pooled,
                            double const* data,
                            size_t width,
                            size_t height,
                            size_t depth,
                            size_t poolSize,
                            size_t poolStride) ;

/* ---------------------------------------------------------------- */
/*                                                 maxPooling (GPU) */
/* ---------------------------------------------------------------- */

template <typename Dtype>
__global__ void maxPooling_gpu_kernel
(const int nthreads, const Dtype* bottom_data,
 const int num, const int channels, const int height,
 const int width, const int pooled_height, const int pooled_width,
 const int ksize, const int stride, Dtype* top_data)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < nthreads) {
    // decode index:
    // index = (n*channels + c)*pooled_height) + ph)*pooled_width + pw
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;
    // pooled patch start and end
    int hstart = ph * stride;
    int hend = min(hstart + ksize, height);
    int wstart = pw * stride;
    int wend = min(wstart + ksize, width);
    Dtype maxval = -FLT_MAX;
    bottom_data += (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        maxval = max(maxval, bottom_data[h * width + w]);
      }
    }
    top_data[index] = maxval;
  }
}

template<typename T>
void maxPooling_gpu(T* pooled,
                    T const* data,
                    size_t width,
                    size_t height,
                    size_t depth,
                    size_t poolSize,
                    size_t poolStride)
{
  int pooledWidth = (width - poolSize)/poolStride + 1 ;
  int pooledHeight = (height - poolSize)/poolStride + 1 ;
  int count = pooledWidth * pooledHeight * depth ;
  maxPooling_gpu_kernel<T><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>
  (count, data, 1, depth, height, width, pooledHeight, pooledWidth, poolSize, poolStride, pooled) ;
}

template
void maxPooling_gpu<float>(float* pooled,
                           float const* data,
                           size_t width,
                           size_t height,
                           size_t depth,
                           size_t poolSize,
                           size_t poolStride) ;

template
void maxPooling_gpu<double>(double* pooled,
                            double const* data,
                            size_t width,
                            size_t height,
                            size_t depth,
                            size_t poolSize,
                            size_t poolStride) ;

/* ---------------------------------------------------------------- */
/*                                         maxPoolingBackward (CPU) */
/* ---------------------------------------------------------------- */

template<typename T>
void maxPoolingBackward_cpu(T* dzdx,
                            T const* data,
                            T const* dzdy,
                            size_t width,
                            size_t height,
                            size_t depth,
                            size_t poolSize,
                            size_t poolStride)
{
  int pooledWidth = (width - poolSize)/poolStride + 1 ;
  int pooledHeight = (height - poolSize)/poolStride + 1 ;

  for (int c = 0; c < depth; ++c) {
    for (int ph = 0; ph < pooledHeight; ++ph) {
      for (int pw = 0; pw < pooledWidth; ++pw) {
        int hstart = ph * poolStride;
        int wstart = pw * poolStride;
        int hend = std::min(hstart + poolSize, height);
        int wend = std::min(wstart + poolSize, width);
        int bestIndex = hstart * width + wstart ;
        T bestValue = data[bestIndex] ;
        for (int h = hstart; h < hend; ++h) {
          for (int w = wstart; w < wend; ++w) {
            int index = h * width + w ;
            T x = data[index] ;
            if (x > bestValue) {
              bestValue = x ;
              bestIndex = index ;
            }
          }
        }
        dzdx[bestIndex] += dzdy[ph * pooledWidth + pw] ;
      }
    }
    data += width*height ;
    dzdx += width*height ;
    dzdy += pooledWidth*pooledHeight ;
  }
}

template
void maxPoolingBackward_cpu<float>(float* dzdx,
                                   float const* data,
                                   float const* dzdy,
                                   size_t width,
                                   size_t height,
                                   size_t depth,
                                   size_t poolSize,
                                   size_t poolStride) ;

template
void maxPoolingBackward_cpu<double>(double* dzdx,
                                    double const* data,
                                    double const* dzdy,
                                    size_t width,
                                    size_t height,
                                    size_t depth,
                                    size_t poolSize,
                                    size_t poolStride) ;

/* ---------------------------------------------------------------- */
/*                                         maxPoolingBackward (GPU) */
/* ---------------------------------------------------------------- */

#if 0
template <typename Dtype>
__global__ void maxPoolingBackward_gpu_kernel
(const int nthreads, const Dtype* bottom_data, const Dtype* top_diff,
 const int num, const int channels, const int height,
 const int width, const int pooled_height, const int pooled_width,
 const int ksize, const int stride, Dtype* bottom_diff)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < nthreads) {
    // find out the local index
    // find out the local offset
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;
    int phstart = (h < ksize) ? 0 : (h - ksize) / stride + 1;
    int phend = min(h / stride + 1, pooled_height);
    int pwstart = (w < ksize) ? 0 : (w - ksize) / stride + 1;
    int pwend = min(w / stride + 1, pooled_width);
    Dtype gradient = 0;
    Dtype bottom_datum =
    bottom_data[((n * channels + c) * height + h) * width + w];
    top_data += (n * channels + c) * pooled_height * pooled_width;
    top_diff += (n * channels + c) * pooled_height * pooled_width;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        gradient += top_diff[ph * pooled_width + pw] *
        (bottom_datum == top_data[ph * pooled_width + pw]);
      }
    }
    bottom_diff[index] = gradient;
  }  // (if index < nthreads)
}
#endif

template <typename Dtype>
__global__ void maxPoolingBackward_gpu_kernel
(const int nthreads, const Dtype* bottom_data, const Dtype* top_diff,
 const int num, const int channels, const int height,
 const int width, const int pooled_height, const int pooled_width,
 const int ksize, const int stride, Dtype* bottom_diff)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < nthreads) {
    // decode index:
    // index = (n*channels + c)*pooled_height) + ph)*pooled_width + pw
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;
    // pooled patch start and end
    int hstart = ph * stride;
    int hend = min(hstart + ksize, height);
    int wstart = pw * stride;
    int wend = min(wstart + ksize, width);
    Dtype bestValue = -FLT_MAX;
    int bestIndex = 0 ;
    bottom_data += (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        int index = h * width + w ;
        Dtype x = bottom_data[index] ;
        if (x > bestValue) {
          bestValue = x ;
          bestIndex = index ;
        }
      }
    }
    bottom_diff[bestIndex] += top_diff[index] ;
  }
}

template<typename T>
void maxPoolingBackward_gpu(T* dzdx,
                    T const* data,
                    T const* dzdy,
                    size_t width,
                    size_t height,
                    size_t depth,
                    size_t poolSize,
                    size_t poolStride)
{
  int pooledWidth = (width - poolSize)/poolStride + 1 ;
  int pooledHeight = (height - poolSize)/poolStride + 1 ;
  int count = pooledWidth * pooledHeight * depth ;
  maxPoolingBackward_gpu_kernel<T><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>
  (count, data, dzdy, 1, depth, height, width, pooledHeight, pooledWidth, poolSize, poolStride, dzdx) ;
}

template
void maxPoolingBackward_gpu<float>(float* pooled,
                                   float const* data,
                                   float const* dzdy,
                                   size_t width,
                                   size_t height,
                                   size_t depth,
                                   size_t poolSize,
                                   size_t poolStride) ;

template
void maxPoolingBackward_gpu<double>(double* pooled,
                                    double const* data,
                                    double const* dzdy,
                                    size_t width,
                                    size_t height,
                                    size_t depth,
                                    size_t poolSize,
                                    size_t poolStride) ;
