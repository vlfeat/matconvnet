/* rip-off convolution from decaf and port it to MATLAB and gpuArrays */

#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cublas_v2.h>
#include <blas.h>
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
  mxGPUArray const *dataGpu ;
  mxGPUArray const *filtersGpu ;
  mxGPUArray *resultGpu ;
  mxGPUArray *tempGpu ;
  mxArray *resultArray ;
  mxArray *tempArray ;

  cublasStatus_t stat;
  cublasHandle_t handle;

  size_t height, width, depth, numImages ;
  size_t filterHeight, filterWidth, filterDepth ;
  size_t numFilters ;
  mwSize dataNumDimensions ;
  mwSize filtersNumDimensions ;
  mwSize const * dataDimensions ;
  mwSize const * filtersDimensions ;
  mwSize resultDimensions [4] ;
  mwSize tempDimensions [3] ;

  bool gpuMode = false ;
  int verbosiy = 1 ;

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
    mexErrMsgTxt("The arguments are not two.") ;
  }

  gpuMode = mxIsGPUArray(in[IN_DATA]) ;

  if (gpuMode) {
    if (!mxIsGPUArray(in[IN_FILTERS])) {
      mexErrMsgTxt("DATA is a GPU array but FILTERS is not.") ;
    }
    dataGpu = mxGPUCreateFromMxArray(in[IN_DATA]) ;
    dataClassID = mxGPUGetClassID(dataGpu) ;
    dataNumDimensions = mxGPUGetNumberOfDimensions(dataGpu) ;
    dataDimensions = mxGPUGetDimensions(dataGpu) ;
    filtersGpu = mxGPUCreateFromMxArray(in[IN_FILTERS]) ;
    filtersClassID = mxGPUGetClassID(filtersGpu) ;
    filtersNumDimensions = mxGPUGetNumberOfDimensions(filtersGpu) ;
    filtersDimensions = mxGPUGetDimensions(filtersGpu) ;
  } else {
    if (mxIsGPUArray(in[IN_FILTERS])) {
      mexErrMsgTxt("DATA is a CPU array but FILTERS is not.") ;
    }
    dataClassID = mxGetClassID(in[IN_DATA]) ;
    dataNumDimensions = mxGetNumberOfDimensions(in[IN_DATA]) ;
    dataDimensions = mxGetDimensions(in[IN_DATA]) ;
    filtersClassID = mxGetClassID(in[IN_FILTERS]) ;
    filtersNumDimensions = mxGetNumberOfDimensions(in[IN_FILTERS]) ;
    filtersDimensions = mxGetDimensions(in[IN_FILTERS]) ;
  }

  if (dataClassID != mxSINGLE_CLASS) {
    mexErrMsgTxt("DATA is not of class SINGLE.");
  }
  if (filtersClassID != mxSINGLE_CLASS) {
    mexErrMsgTxt("FILTERS is not of class SINGLE.");
  }

  height = dataDimensions[0] ;
  width = dataDimensions[1] ;
  switch (dataNumDimensions) {
  case 2 : depth = 1 ; numImages = 1 ; break ;
  case 3 : depth = dataDimensions[2] ; numImages = 1 ; break ;
  case 4 : depth = dataDimensions[2] ; numImages = dataDimensions[3] ; break ;
  default:  mexErrMsgTxt("DATA has neither two or three dimensions.") ; break ;
  }

  filterHeight = filtersDimensions[0] ;
  filterWidth = filtersDimensions[1] ;
  switch (filtersNumDimensions) {
  case 2 : filterDepth = 1 ; numFilters = 1 ; break ;
  case 3 : filterDepth = filtersDimensions[2] ; numFilters = 1 ; break ;
  case 4 : filterDepth = filtersDimensions[2] ; numFilters = filtersDimensions[3] ; break ;
  default:  mexErrMsgTxt("FILTERS has neither two, three, nor four dimensions.") ; break ;
  }

  if (filterWidth != filterHeight) {
    mexErrMsgTxt("Non-square FILTERS not supported yet.") ;
  }

  resultDimensions[0] = height - filterHeight + 1 ;
  resultDimensions[1] = width - filterWidth + 1 ;
  resultDimensions[2] = numFilters ;
  resultDimensions[3] = numImages ;

  tempDimensions[0] = height - filterHeight + 1 ;
  tempDimensions[1] = width - filterWidth + 1 ;
  tempDimensions[2] = filterHeight*filterWidth*filterDepth ;

  if (verbosiy > 0) {
    mexPrintf("gconv: mode %s\n", gpuMode?"gpu":"cpu") ;
    mexPrintf("gconv: data: %d x %d x %d x %d [%.1f MB]\n", height, width, depth, numImages,
              (double)(height*width*depth*numImages*4)/1024.0/1024.0) ;
    mexPrintf("gconv: filters: %d x %d x %d x %d [%.1f MB]\n", filterHeight, filterWidth, filterDepth, numFilters,
              (double)(resultDimensions[0]*resultDimensions[1]*resultDimensions[2]*resultDimensions[3]*4)/1024.0/1024.0) ;
    mexPrintf("gconv: result: %d x %d x %d x %d [%.1f MB]\n", resultDimensions[0], resultDimensions[1], resultDimensions[2], resultDimensions[3],
              (double)(resultDimensions[0]*resultDimensions[1]*resultDimensions[2]*resultDimensions[3]*4)/1024.0/1024.0) ;
    mexPrintf("gconv: temp: %d x %d x %d [%.1f MB]\n", tempDimensions[0], tempDimensions[1], tempDimensions[2],
              (double)(tempDimensions[0]*tempDimensions[1]*tempDimensions[2]*4)/1024.0/1024.0) ;
  }

  if (depth != filterDepth) {
    mexErrMsgTxt("DATA and FILTERS dimensions do not match.") ;
  }

  if (height < filterHeight ||  width < filterWidth) {
    mexErrMsgTxt("FILTERS are larger than the DATA.") ;
  }

  if (filterHeight == 0 || filterWidth == 0 || filterDepth == 0) {
    mexErrMsgTxt("A dimension of FILTERS is void.") ;
  }

  /* -------------------------------------------------------------- */
  /*                                                    Do the work */
  /* -------------------------------------------------------------- */
  // im2col should be called im2row

  if (gpuMode) {
    tempGpu = mxGPUCreateGPUArray(3, tempDimensions,
                                  mxSINGLE_CLASS,
                                  mxREAL,
                                  MX_GPU_DO_NOT_INITIALIZE) ;
    resultGpu = mxGPUCreateGPUArray(4, resultDimensions,
                                    mxSINGLE_CLASS,
                                    mxREAL,
                                    MX_GPU_DO_NOT_INITIALIZE) ;
  } else {
    tempArray = mxCreateNumericArray(3, tempDimensions,
                                     mxSINGLE_CLASS,
                                     mxREAL) ;
    resultArray = mxCreateNumericArray(4, resultDimensions,
                                       mxSINGLE_CLASS,
                                       mxREAL) ;
  }

  for (int image = 0 ; image < numImages ; ++image) {
    float alpha = 1 ;
    float beta = 0 ;
    char opA = 't' ;
    char opB = 'n' ;
    ptrdiff_t m = resultDimensions[0] * resultDimensions[1] ;
    ptrdiff_t n = numFilters ;
    ptrdiff_t k = filterHeight*filterWidth*filterDepth ;
    ptrdiff_t dataOffset = (width*height*depth) * image ;
    ptrdiff_t resultOffset = (resultDimensions[0]*resultDimensions[1]*resultDimensions[2]) * image ;

    if (gpuMode) {
#if 1
      im2col_gpu<float>((float const*)mxGPUGetDataReadOnly(dataGpu) + dataOffset,
                        depth, height, width,
                        filterHeight,
                        1, // stride,
                        (float *)mxGPUGetData(tempGpu)) ;
#endif

      // op = N (not transposed), T (transposed)
      // C <- alpha op(A)op(B) + beta C
      // A is m x k, B is k x n and C is m x n.
#if 1
      cublasSgemm(handle,
                  (opA == 'n') ? CUBLAS_OP_N : CUBLAS_OP_T,
                  (opB == 'n') ? CUBLAS_OP_N : CUBLAS_OP_T,
                  (int)m, (int)n, (int)k,
                  &alpha,
                  (float const*)mxGPUGetDataReadOnly(tempGpu), (opA == 'n') ? (int)m : (int)k,
                  (float const*)mxGPUGetDataReadOnly(filtersGpu), (opB == 'n') ? (int)k : (int)n,
                  &beta,
                  (float*)mxGPUGetData(resultGpu) + resultOffset, (int)m) ;
#endif
    } else {
#if 1
      im2col_cpu<float>((float const*)mxGetData(in[IN_DATA]) + dataOffset,
                        depth, height, width,
                        filterHeight,
                        1, // stride,
                        (float *)mxGetData(tempArray)) ;
#endif

#if 1
      sgemm(&opA, &opB,
            &m, &n, &k,
            &alpha,
            (float*)mxGetData(tempArray), (opA == 'n') ? &m : &k,
            (float*)mxGetData(in[IN_FILTERS]),(opB == 'n') ? &k : &n,
            &beta,
            (float*)mxGetData(resultArray) + resultOffset, &m) ;
#endif
    }
  }

  if (gpuMode ) {
    cublasDestroy(handle);
    out[OUT_RESULT] = mxGPUCreateMxArrayOnGPU(resultGpu) ;
    mxGPUDestroyGPUArray(dataGpu) ;
    mxGPUDestroyGPUArray(filtersGpu) ;
    mxGPUDestroyGPUArray(resultGpu) ;
    mxGPUDestroyGPUArray(tempGpu) ;
  } else {
    mxDestroyArray(tempArray);
    out[OUT_RESULT] = resultArray ;
  }
}