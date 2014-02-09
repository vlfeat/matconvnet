/* rip-off convolution from decaf and port it to MATLAB and gpuArrays */

#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <iostream>

#define FATAL std::cout
#define LOG(x) x

#define CUDA_POST_KERNEL_CHECK \
  if (cudaSuccess != cudaPeekAtLastError()) \
    LOG(FATAL) << "Cuda kernel failed. Error: " \
        << cudaGetErrorString(cudaPeekAtLastError())

// We will use 1024 threads per block, which requires cuda sm_2x or above.
#if __CUDA_ARCH__ >= 200
    const int CAFFE_CUDA_NUM_THREADS = 1024;
#else
    const int CAFFE_CUDA_NUM_THREADS = 512;
#endif


inline int CAFFE_GET_BLOCKS(const int N) {
  return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
}

/* -------------------------------------------------------------------- */
/*                                                          CPU variant */
/* -------------------------------------------------------------------- */

template <typename Dtype>
void im2col_cpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int ksize, const int stride,
    Dtype* data_col) {
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

// Explicit instantiation
template void im2col_cpu<float>(const float* data_im, const int channels,
    const int height, const int width, const int ksize, const int stride,
    float* data_col);
template void im2col_cpu<double>(const double* data_im, const int channels,
    const int height, const int width, const int ksize, const int stride,
    double* data_col);


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

// Explicit instantiation
template void col2im_cpu<float>(const float* data_col, const int channels,
    const int height, const int width, const int psize, const int stride,
    float* data_im);
template void col2im_cpu<double>(const double* data_col, const int channels,
    const int height, const int width, const int psize, const int stride,
    double* data_im);

/* -------------------------------------------------------------------- */
/*                                                          GPU variant */
/* -------------------------------------------------------------------- */

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
  im2col_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
    num_kernels, data_im, height, width, ksize, stride, height_col, width_col,
    data_col);
  CUDA_POST_KERNEL_CHECK;
}

// Explicit instantiation
template void im2col_gpu<float>(const float* data_im, const int channels,
  const int height, const int width, const int ksize, const int stride,
    float* data_col);
template void im2col_gpu<double>(const double* data_im, const int channels,
    const int height, const int width, const int ksize, const int stride,
    double* data_col);

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
  col2im_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, data_col, height, width, channels, ksize, stride,
      height_col, width_col, data_im);
  CUDA_POST_KERNEL_CHECK;
}


// Explicit instantiation
template void col2im_gpu<float>(const float* data_col, const int channels,
    const int height, const int width, const int psize, const int stride,
    float* data_im);
template void col2im_gpu<double>(const double* data_col, const int channels,
    const int height, const int width, const int psize, const int stride,
    double* data_im);


void convolution_gpu(mxGPUArray *A, mxGPUArray *B, mxGPUArray *C)
{
#if 0
  d_A = (double const *)(mxGPUGetDataReadOnly(A));

  d_B = (double *)(mxGPUGetData(B));

  int const threadsPerBlock = 256;
  int blocksPerGrid;

  /*
   * Call the kernel using the CUDA runtime API. We are using a 1-d grid here,
   * and it would be possible for the number of elements to be too large for
   * the grid. For this example we are not guarding against this possibility.
   */
  N = (int)(mxGPUGetNumberOfElements(A));
  blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  TimesTwo<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, N);

  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  Dtype* col_data = col_buffer_.mutable_gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  int weight_offset = M_ * K_;
  int col_offset = K_ * N_;
  int top_offset = M_ * N_;
  for (int n = 0; n < NUM_; ++n) {
    // First, im2col
    im2col_gpu(bottom_data + bottom[0]->offset(n), CHANNELS_, HEIGHT_,
        WIDTH_, KSIZE_, STRIDE_, col_data);
    // Second, innerproduct with groups
    for (int g = 0; g < GROUP_; ++g) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_,
        (Dtype)1., weight + weight_offset * g, col_data + col_offset * g,
        (Dtype)0., top_data + (*top)[0]->offset(n) + top_offset * g);
    }
    // third, add bias
    if (biasterm_) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, NUM_OUTPUT_,
          N_, 1, (Dtype)1., this->blobs_[1]->gpu_data(),
          reinterpret_cast<const Dtype*>(bias_multiplier_->gpu_data()),
          (Dtype)1., top_data + (*top)[0]->offset(n));
    }
  }
#endif
}

/* ---------------------------------------------------------------- */
/*                                                         Do stuff */
/* ---------------------------------------------------------------- */

enum {
  IN_A = 0, IN_B, IN_END
} ;

enum {
  OUT_C = 0, OUT_END
} ;

void mexFunction(int nout, mxArray *out[],
                 int nin, mxArray const *in[])
{
  mxGPUArray const *A ;
  mxGPUArray const *B ;
  mxGPUArray *C ;

  size_t height, width, dimension ;
  size_t filterHeight, filterWidth, filterDimension ;
  size_t numFilters ;
  mwSize resultDimensions [3] ;
  mwSize const * ADimensions ;
  mwSize const * BDimensions ;

  /* Initialize the MathWorks GPU API. */
  mxInitGPU();

  /* -------------------------------------------------------------- */
  /*                                            Check the arguments */
  /* -------------------------------------------------------------- */

  /* Throw an error if the input is not a GPU array. */
  if (nin != 2) {
    mexErrMsgTxt("Other than two arguments provided.") ;
  }
  if (!mxIsGPUArray(in[IN_A])) {
    mexErrMsgTxt("A is not a GPU array.") ;
  }
  if (!mxIsGPUArray(in[IN_B])) {
    mexErrMsgTxt("B is not a GPU array.") ;
  }

  A = mxGPUCreateFromMxArray(in[IN_A]) ;
  B = mxGPUCreateFromMxArray(in[IN_B]) ;

  if (mxGPUGetClassID(A) != mxSINGLE_CLASS) {
    mexErrMsgTxt("A is not of class SINGLE.");
  }
  if (mxGPUGetClassID(B) != mxSINGLE_CLASS) {
    mexErrMsgTxt("B is not of class SINGLE.");
  }

  ADimensions = mxGPUGetDimensions(A) ;
  height = ADimensions[0] ;
  width = ADimensions[1] ;
  mexPrintf("%d %d %d\n", width, mxGPUGetNumberOfDimensions(A), sizeof(mwSize)) ;
  switch (mxGPUGetNumberOfDimensions(A)) {
  case 2 : dimension = 1 ; break ;
  case 3 : dimension = ADimensions[2] ; break ;
  default:  mexErrMsgTxt("A has neither two or three dimensions.") ; break ;
  }
  //mxFree(ADimensions) ;

  BDimensions = mxGPUGetDimensions(B) ;
  filterHeight = BDimensions[0] ;
  filterWidth = BDimensions[1] ;
  switch (mxGPUGetNumberOfDimensions(B)) {
  case 2 : filterDimension = 1 ; numFilters = 1 ; break ;
  case 3 : filterDimension = BDimensions[2] ; numFilters = 1 ; break ;
  case 4 : filterDimension = BDimensions[2] ; numFilters = BDimensions[3] ; break ;
  default:  mexErrMsgTxt("B has neither two, three, nor four dimensions.") ; break ;
  }
  //mxFree(BDimensions) ;

  /*
  resultDimensions[0] = height - filterHeight + 1 ;
  resultDimensions[1] = width - filterWidth + 1 ;
  resultDimensions[2] = numFilters ;
  */

  resultDimensions[0] = height - filterHeight + 1 ;
  resultDimensions[1] = width - filterWidth + 1 ;
  resultDimensions[2] = filterHeight*filterWidth*dimension ;

  mexPrintf("A: %d x %d x %d\n", height, width, dimension) ;
  mexPrintf("B: %d x %d x %d x %d\n", filterHeight, filterWidth, filterDimension, numFilters) ;
  mexPrintf("C: %d x %d x %d\n", resultDimensions[0], resultDimensions[1], resultDimensions[2]) ;

  if (dimension != filterDimension) {
    mexErrMsgTxt("A and B dimensions do not match.") ;
  }

  if (height < filterHeight ||  width < filterWidth) {
    mexErrMsgTxt("Filters are larger than the image.") ;
  }

  if (filterHeight == 0 || filterWidth == 0 || filterDimension == 0) {
    mexErrMsgTxt("A dimension of B is void.") ;
  }

  C = mxGPUCreateGPUArray(3, resultDimensions,
                          mxSINGLE_CLASS,
                          mxREAL,
                          MX_GPU_DO_NOT_INITIALIZE) ;

  im2col_gpu<float>((float const*)mxGPUGetDataReadOnly(A),
                    dimension, height, width,
                    filterHeight, // filter size
                    1, // stride,
                    (float *)mxGPUGetData(C)) ;

  //  void im2col_gpu(const Dtype* data_im, const int channels,
  //               const int height, const int width, const int ksize, const int stride,
  //               Dtype* data_col) ;

  out[OUT_C] = mxGPUCreateMxArrayOnGPU(C) ;
  mxGPUDestroyGPUArray(A) ;
  mxGPUDestroyGPUArray(B) ;
  mxGPUDestroyGPUArray(C) ;
}
