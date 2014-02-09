/* rip-off convolution from decaf and port it to MATLAB and gpuArrays */

#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cublas_v2.h>
#include <iostream>


// Hack caffe away

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

// place copies of filter-shaped volumes as rows of a matrix
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

/* ---------------------------------------------------------------- */
/*                                                         Do stuff */
/* ---------------------------------------------------------------- */

enum {
  IN_DATA = 0, IN_FILTERS, IN_END
} ;

enum {
  OUT_RESULT = 0, OUT_END
} ;

void mexFunction(int nout, mxArray *out[],
                 int nin, mxArray const *in[])
{
  mxClassID dataClassID ;
  mxClassID filtersClassID ;
  mxGPUArray const *data ;
  mxGPUArray const *filters ;
  mxGPUArray *result ;
  mxGPUArray *temp ;

  cublasStatus_t stat;
  cublasHandle_t handle;

  size_t height, width, dimension ;
  size_t filterHeight, filterWidth, filterDimension ;
  size_t numFilters ;
  mwSize const * dataDimensions ;
  mwSize const * filtersDimensions ;
  mwSize resultDimensions [3] ;
  mwSize tempDimensions [3] ;

  bool gpuMode = false ;

  /* Initialize the MathWorks GPU API. */
  mxInitGPU() ;

  stat = cublasCreate(&handle);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    mexErrMsgTxt("Could not initialize cuBLAS.") ;
  }

  /* -------------------------------------------------------------- */
  /*                                            Check the arguments */
  /* -------------------------------------------------------------- */

  /* Throw an error if the input is not a GPU array. */
  if (nin != 2) {
    mexErrMsgTxt("Other than two arguments provided.") ;
  }

  gpuMode = mxIsGPUArray(in[IN_DATA]) ;

  if (gpuMode) {
    if (!mxIsGPUArray(in[IN_FILTERS])) {
      mexErrMsgTxt("DATA is a GPU array but FILTERS is not.") ;
    }
    data = mxGPUCreateFromMxArray(in[IN_DATA]) ;
    dataClassID = mxGPUGetClassID(data) ;
    filters = mxGPUCreateFromMxArray(in[IN_FILTERS]) ;
    filtersClassID = mxGPUGetClassID(filters) ;
  } else {
    if (!mxIsGPUArray(in[IN_FILTERS])) {
      mexErrMsgTxt("DATA is a CPU array but FILTERS is not.") ;
    }
    dataClassID = mxGetClassID(in[IN_DATA]) ;
    filtersClassID = mxGetClassID(in[IN_FILTERS]) ;
  }

  if (dataClassID != mxSINGLE_CLASS) {
    mexErrMsgTxt("DATA is not of class SINGLE.");
  }
  if (filtersClassID != mxSINGLE_CLASS) {
    mexErrMsgTxt("FILTERS is not of class SINGLE.");
  }

  dataDimensions = mxGPUGetDimensions(data) ;
  height = dataDimensions[0] ;
  width = dataDimensions[1] ;
  switch (mxGPUGetNumberOfDimensions(data)) {
  case 2 : dimension = 1 ; break ;
  case 3 : dimension = dataDimensions[2] ; break ;
  default:  mexErrMsgTxt("DATA has neither two or three dimensions.") ; break ;
  }
  //mxFree(DataDimensions) ;

  filtersDimensions = mxGPUGetDimensions(filters) ;
  filterHeight = filtersDimensions[0] ;
  filterWidth = filtersDimensions[1] ;
  switch (mxGPUGetNumberOfDimensions(filters)) {
  case 2 : filterDimension = 1 ; numFilters = 1 ; break ;
  case 3 : filterDimension = filtersDimensions[2] ; numFilters = 1 ; break ;
  case 4 : filterDimension = filtersDimensions[2] ; numFilters = filtersDimensions[3] ; break ;
  default:  mexErrMsgTxt("FILTERS has neither two, three, nor four dimensions.") ; break ;
  }
  //mxFree(FiltersDimensions) ;

  if (filterWidth != filterHeight) {
    mexErrMsgTxt("Non-square FILTERS not supported yet.") ;
  }

  resultDimensions[0] = height - filterHeight + 1 ;
  resultDimensions[1] = width - filterWidth + 1 ;
  resultDimensions[2] = numFilters ;

  tempDimensions[0] = height - filterHeight + 1 ;
  tempDimensions[1] = width - filterWidth + 1 ;
  tempDimensions[2] = filterHeight*filterWidth*dimension ;

  mexPrintf("data: %d x %d x %d\n", height, width, dimension) ;
  mexPrintf("filters: %d x %d x %d x %d\n", filterHeight, filterWidth, filterDimension, numFilters) ;
  mexPrintf("result: %d x %d x %d\n", resultDimensions[0], resultDimensions[1], resultDimensions[2]) ;
  mexPrintf("temp: %d x %d x %d\n", tempDimensions[0], tempDimensions[1], tempDimensions[2]) ;

  if (dimension != filterDimension) {
    mexErrMsgTxt("DATA and FILTERS dimensions do not match.") ;
  }

  if (height < filterHeight ||  width < filterWidth) {
    mexErrMsgTxt("FILTERS are larger than the DATA.") ;
  }

  if (filterHeight == 0 || filterWidth == 0 || filterDimension == 0) {
    mexErrMsgTxt("A dimension of FILTERS is void.") ;
  }

  temp = mxGPUCreateGPUArray(3, tempDimensions,
                             mxSINGLE_CLASS,
                             mxREAL,
                             //MX_GPU_INITIALIZE_VALUES);
                             MX_GPU_DO_NOT_INITIALIZE) ;

#if 1
  // contrary to the name, this is im2row ... sigh
  im2col_gpu<float>((float const*)mxGPUGetDataReadOnly(data),
                    dimension, height, width,
                    filterHeight, // filter size
                    1, // stride,
                    (float *)mxGPUGetData(temp)) ;
#endif
 
  result = mxGPUCreateGPUArray(3, resultDimensions,
                               mxSINGLE_CLASS,
                               mxREAL,
                               //MX_GPU_INITIALIZE_VALUES);
                               MX_GPU_DO_NOT_INITIALIZE) ;

#if 1
  {
    float alpha = 1 ;
    float beta = 0 ;
    cublasSgemm
      (handle,
       CUBLAS_OP_N, // not transposed
       CUBLAS_OP_N, // not transposed
       resultDimensions[0] * resultDimensions[1], // m: op(B) cols [= result cols]
       numFilters, // n: op(A) rows [= results rows]
       filterHeight*filterWidth*filterDimension, // k: op(A) cols = op(B) rows [= dot prod length]
       &alpha,
       (float const*)mxGPUGetDataReadOnly(temp), // A: im2col output
       resultDimensions[0] * resultDimensions[1], // A: leading dimension
       (float const*)mxGPUGetDataReadOnly(filters), // B: filters
       filterHeight*filterWidth*filterDimension, // B: leading dimension
       &beta,
       (float*)mxGPUGetData(result), // C: output image
       resultDimensions[0] * resultDimensions[1] // C: leading dimension
       ) ;
  }
#endif

  cublasDestroy(handle);
  out[OUT_RESULT] = mxGPUCreateMxArrayOnGPU(result) ;
  mxGPUDestroyGPUArray(data) ;
  mxGPUDestroyGPUArray(filters) ;
  mxGPUDestroyGPUArray(result) ;
  mxGPUDestroyGPUArray(temp) ;
}
