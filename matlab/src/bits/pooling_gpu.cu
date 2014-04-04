/** @file pooling_gpu.cu
 ** @brief Max pooling filters (GPU)
 ** @author Andrea Vedaldi
 **/

#include "im2col.cpp"
#include "gpu.hpp"
#include <float.h>
#include <sm_20_atomic_functions.h>

/* ---------------------------------------------------------------- */
/*                                                 maxPooling (GPU) */
/* ---------------------------------------------------------------- */

template <typename Dtype>
__global__ void maxPooling_gpu_kernel
(const int nthreads, const Dtype* bottom_data,
 const int num, const int channels, const int height,
 const int width, const int pooled_height, const int pooled_width,
 const int ksize, const int stride, const int pad, Dtype* top_data)
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
    int wstart = max(pw * stride - pad, 0) ;
    int hstart = max(ph * stride - pad, 0) ;
    int wend = min(pw * stride - pad + ksize, width) ;
    int hend = min(ph * stride - pad + ksize, height) ;
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
                    size_t stride,
                    size_t pad)
{
  int pooledWidth = (width + 2*pad - poolSize)/stride + 1 ;
  int pooledHeight = (height + 2*pad - poolSize)/stride + 1 ;
  int count = pooledWidth * pooledHeight * depth ;
  maxPooling_gpu_kernel<T><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>
  (count, data, 1, depth, height, width, pooledHeight, pooledWidth, poolSize, stride, pad, pooled) ;
}

template
void maxPooling_gpu<float>(float* pooled,
                           float const* data,
                           size_t width,
                           size_t height,
                           size_t depth,
                           size_t poolSize,
                           size_t stride,
                           size_t pad) ;

template
void maxPooling_gpu<double>(double* pooled,
                            double const* data,
                            size_t width,
                            size_t height,
                            size_t depth,
                            size_t poolSize,
                            size_t stride,
                            size_t pad) ;

/* ---------------------------------------------------------------- */
/*                                         maxPoolingBackward (GPU) */
/* ---------------------------------------------------------------- */

template <typename Dtype>
__global__ void maxPoolingBackward_gpu_kernel
(const int nthreads, const Dtype* bottom_data, const Dtype* top_diff,
 const int num, const int channels, const int height,
 const int width, const int pooled_height, const int pooled_width,
 const int ksize, const int stride, const int pad,
 Dtype* bottom_diff)
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
    int wstart = max(pw * stride - pad, 0) ;
    int hstart = max(ph * stride - pad, 0) ;
    int wend = min(pw * stride - pad + ksize, width) ;
    int hend = min(ph * stride - pad + ksize, height) ;
    Dtype bestValue = -FLT_MAX;
    int bestIndex = 0 ;
    bottom_data += (n * channels + c) * height * width;
    bottom_diff += (n * channels + c) * height * width;
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
    /*
     This is bad, but required to eliminate a race condition when writing
     to bottom_diff.
     Caffe goes the other way around, but requrires remembering the layer
     output, or the maximal indexes.
     */
    atomicAdd(bottom_diff + bestIndex, top_diff[index]) ;
    //bottom_diff[bestIndex] += top_diff[index] ;
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
                            size_t stride,
                            size_t pad)
{
  int pooledWidth = (width + 2*pad - poolSize)/stride + 1 ;
  int pooledHeight = (height + 2*pad - poolSize)/stride + 1 ;
  int count = pooledWidth * pooledHeight * depth ;
  maxPoolingBackward_gpu_kernel<T><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>
  (count, data, dzdy, 1, depth, height, width, pooledHeight, pooledWidth, poolSize, stride, pad, dzdx) ;
}

template
void maxPoolingBackward_gpu<float>(float* pooled,
                                   float const* data,
                                   float const* dzdy,
                                   size_t width,
                                   size_t height,
                                   size_t depth,
                                   size_t poolSize,
                                   size_t stride,
                                   size_t pad) ;

#if 0
template
void maxPoolingBackward_gpu<double>(double* pooled,
                                    double const* data,
                                    double const* dzdy,
                                    size_t width,
                                    size_t height,
                                    size_t depth,
                                    size_t poolSize,
                                    size_t stride,
                                    size_t pad) ;
#endif
