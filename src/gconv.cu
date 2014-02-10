/* rip-off convolution from decaf and port it to MATLAB and gpuArrays */

#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cublas_v2.h>
#include <blas.h>
#include <iostream>

#include "bits/im2col.cpp"

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
    mxInitGPU() ;
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
      mexErrMsgTxt("Could not initialize cuBLAS.") ;
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
    double const MB = 1024.0*1024.0 ;
    mexPrintf("gconv: mode %s\n", gpuMode?"gpu":"cpu") ;
    mexPrintf("gconv: data: %d x %d x %d x %d [%.1f MB]\n",
              height, width, depth, numImages,
              (double)(height*width*depth*numImages*4)/MB) ;
    mexPrintf("gconv: filters: %d x %d x %d x %d [%.1f MB]\n",
              filterHeight, filterWidth, filterDepth, numFilters,
              (double)(filterHeight*filterWidth*filterDepth*numFilters*4)/MB) ;
    mexPrintf("gconv: result: %d x %d x %d x %d [%.1f MB]\n",
              resultDimensions[0], resultDimensions[1], resultDimensions[2], resultDimensions[3],
              (double)(resultDimensions[0]*resultDimensions[1]*resultDimensions[2]*resultDimensions[3]*4)/MB) ;
    mexPrintf("gconv: temp: %d x %d x %d [%.1f MB]\n",
              tempDimensions[0], tempDimensions[1], tempDimensions[2],
              (double)(tempDimensions[0]*tempDimensions[1]*tempDimensions[2]*4)/MB) ;
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
    char opA = 'n' ;
    char opB = 'n' ;
    ptrdiff_t m = resultDimensions[0]*resultDimensions[1] ;
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
#if 1
      // op = N (not transposed), T (transposed)
      // C <- alpha op(A)op(B) + beta C
      // A is m x k, B is k x n and C is m x n.
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
      if (opA == 't') {
        im2row_cpu<float>((float const*)mxGetData(in[IN_DATA]) + dataOffset,
                          depth, height, width,
                          filterHeight,
                          1, // stride,
                          (float *)mxGetData(tempArray)) ;
      } else {
        im2col_cpu<float>((float const*)mxGetData(in[IN_DATA]) + dataOffset,
                          depth, height, width,
                          filterHeight,
                          1, // stride,
                          (float *)mxGetData(tempArray)) ;
      }
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

  if (gpuMode) {
    out[OUT_RESULT] = mxGPUCreateMxArrayOnGPU(resultGpu) ;
    mxGPUDestroyGPUArray(dataGpu) ;
    mxGPUDestroyGPUArray(filtersGpu) ;
    mxGPUDestroyGPUArray(resultGpu) ;
    mxGPUDestroyGPUArray(tempGpu) ;
    cublasDestroy(handle);
  } else {
    mxDestroyArray(tempArray);
    out[OUT_RESULT] = resultArray ;
  }
}