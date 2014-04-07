/** @file gconv.cu
 ** @brief Convolution block
 ** @author Andrea Vedaldi
 **/

#include "mex.h"
#ifdef ENABLE_GPU
#include "gpu/mxGPUArray.h"
#endif
#include "bits/mexutils.h"
#include "bits/im2col.hpp"

#include <blas.h>
#include <iostream>
#include <assert.h>

#ifdef ENABLE_GPU
#include <cublas_v2.h>
#endif

/* option codes */
enum {
  opt_stride = 0,
  opt_pad,
  opt_verbose
} ;

/* options */
vlmxOption  options [] = {
  {"Stride",           1,   opt_stride            },
  {"Pad",              1,   opt_pad               },
  {"Verbose",          0,   opt_verbose           },
  {0,                  0,   0                     }
} ;

enum {
  IN_DATA = 0, IN_FILTERS, IN_DER, IN_END
} ;

enum {
  OUT_RESULT = 0, OUT_RESULT2, OUT_END
} ;

void mexFunction(int nout, mxArray *out[],
                 int nin, mxArray const *in[])
{
  mxClassID dataClassID ;
  mxClassID filtersClassID ;
  mxClassID derClassID ;

  mxArray *resultArray ;
  mxArray *dfiltersArray ;
  mxArray *tempArray ;

  size_t height, width, depth, numImages ;
  size_t filterHeight, filterWidth, filterDepth, numFilters ;
  size_t derHeight, derWidth, derDepth, numDerImages ;
  int stride = 1 ;
  int pad = 0 ;
  int numGroups = 1 ;
  mwSize dataNumDimensions ;
  mwSize filtersNumDimensions ;
  mwSize derNumDimensions ;
  mwSize const * dataDimensions ;
  mwSize const * filtersDimensions ;
  mwSize const * derDimensions ;
  mwSize resultDimensions [4] ;
  mwSize dfiltersDimensions [4] ;
  mwSize tempDimensions [3] ;

#if ENABLE_GPU
  mxGPUArray const *dataGpu ;
  mxGPUArray const *filtersGpu ;
  mxGPUArray const *derGpu ;
  mxGPUArray *resultGpu ;
  mxGPUArray *dfiltersGpu ;
  mxGPUArray *tempGpu ;
  cublasStatus_t stat;
  cublasHandle_t handle;
  bool gpuMode = false ;
#else
  bool const gpuMode = false ;
#endif
  bool backMode = false ;

  int verbosity = 0 ;
  int opt ;
  int next = IN_END ;
  mxArray const *optarg ;

  /* -------------------------------------------------------------- */
  /*                                            Check the arguments */
  /* -------------------------------------------------------------- */

  if (nin < 2) {
    mexErrMsgTxt("The arguments are less than two.") ;
  }

  if (nin > 2 && vlmxIsString(in[2],-1)) {
    next = 2 ;
    backMode = 0 ;
  } else {
    backMode = (nin >= 3) ;
  }

  while ((opt = vlmxNextOption (in, nin, options, &next, &optarg)) >= 0) {
    switch (opt) {
      case opt_verbose :
        ++ verbosity ;
        break ;

      case opt_stride :
        if (!vlmxIsPlainScalar(optarg) || (stride = (int) *mxGetPr(optarg)) < 1) {
          mexErrMsgTxt("STRIDE must be a positive integer.") ;
        }
        break ;

      case opt_pad :
        if (!vlmxIsPlainScalar(optarg) || (pad = (int) *mxGetPr(optarg)) < 0) {
          mexErrMsgTxt("PAD must be a non-negative integer.") ;
        }
        break ;

      default: break ;
    }
  }

#if ENABLE_GPU
  gpuMode = mxIsGPUArray(in[IN_DATA]) ;
#else
  if (!mxIsNumeric(in[IN_DATA])) {
    mexErrMsgTxt("DATA must be numeric (note: GPU support not compiled).") ;
  }
#endif

  if (gpuMode) {
#ifdef ENABLE_GPU
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
    if (backMode) {
      if (!mxIsGPUArray(in[IN_DER])) {
        mexErrMsgTxt("DATA is a GPU array but FILTERS is not.") ;
      }
      derGpu = mxGPUCreateFromMxArray(in[IN_DER]) ;
      derClassID = mxGPUGetClassID(derGpu) ;
      derNumDimensions = mxGPUGetNumberOfDimensions(derGpu) ;
      derDimensions = mxGPUGetDimensions(derGpu) ;
    }
#else
    assert(false) ;
#endif
  } else {
    if (!mxIsNumeric(in[IN_FILTERS])) {
      mexErrMsgTxt("DATA is a CPU array but FILTERS is not.") ;
    }
    dataClassID = mxGetClassID(in[IN_DATA]) ;
    dataNumDimensions = mxGetNumberOfDimensions(in[IN_DATA]) ;
    dataDimensions = mxGetDimensions(in[IN_DATA]) ;
    filtersClassID = mxGetClassID(in[IN_FILTERS]) ;
    filtersNumDimensions = mxGetNumberOfDimensions(in[IN_FILTERS]) ;
    filtersDimensions = mxGetDimensions(in[IN_FILTERS]) ;
    if (backMode) {
      derClassID = mxGetClassID(in[IN_DER]) ;
      derNumDimensions = mxGetNumberOfDimensions(in[IN_DER]) ;
      derDimensions = mxGetDimensions(in[IN_DER]) ;
    }
  }

  if (dataClassID != mxSINGLE_CLASS) {
    mexErrMsgTxt("DATA is not of class SINGLE.");
  }
  if (filtersClassID != mxSINGLE_CLASS) {
    mexErrMsgTxt("FILTERS is not of class SINGLE.");
  }
  if (backMode && (derClassID != mxSINGLE_CLASS)) {
    mexErrMsgTxt("DER is not of class SINGLE.");
  }

  height = dataDimensions[0] ;
  width = dataDimensions[1] ;
  switch (dataNumDimensions) {
    case 2 : depth = 1 ; numImages = 1 ; break ;
    case 3 : depth = dataDimensions[2] ; numImages = 1 ; break ;
    case 4 : depth = dataDimensions[2] ; numImages = dataDimensions[3] ; break ;
    default:  mexErrMsgTxt("DATA has neither two nor three dimensions.") ; break ;
  }

  filterHeight = filtersDimensions[0] ;
  filterWidth = filtersDimensions[1] ;
  switch (filtersNumDimensions) {
    case 2 : filterDepth = 1 ; numFilters = 1 ; break ;
    case 3 : filterDepth = filtersDimensions[2] ; numFilters = 1 ; break ;
    case 4 : filterDepth = filtersDimensions[2] ; numFilters = filtersDimensions[3] ; break ;
    default:  mexErrMsgTxt("FILTERS has neither two, three, nor four dimensions.") ; break ;
  }

  if (backMode) {
    derHeight = derDimensions[0] ;
    derWidth = derDimensions[1] ;
    switch (derNumDimensions) {
      case 2 : derDepth = 1 ; numDerImages = 1 ; break ;
      case 3 : derDepth = derDimensions[2] ; numDerImages = 1 ; break ;
      case 4 : derDepth = derDimensions[2] ; numDerImages = derDimensions[3] ; break ;
      default:  mexErrMsgTxt("DER has neither two, three, nor four dimensions.") ; break ;
    }
  }

  if (filterWidth != filterHeight) {
    mexErrMsgTxt("Non-square FILTERS not supported yet.") ;
  }

  /* grouped filters */
  numGroups = depth / filterDepth ;

  if (!backMode) {
    resultDimensions[0] = (height + 2*pad - filterHeight)/stride + 1 ;
    resultDimensions[1] = (width + 2*pad - filterHeight)/stride + 1 ;
    resultDimensions[2] = numFilters ;
    resultDimensions[3] = numImages ;
  } else {
    resultDimensions[0] = height ;
    resultDimensions[1] = width ;
    resultDimensions[2] = depth ;
    resultDimensions[3] = numImages ;
    dfiltersDimensions[0] = filterHeight ;
    dfiltersDimensions[1] = filterWidth ;
    dfiltersDimensions[2] = filterDepth ;
    dfiltersDimensions[3] = numFilters ;
  }

  tempDimensions[0] = (height + 2*pad - filterHeight)/stride + 1 ;
  tempDimensions[1] = (width + 2*pad - filterHeight)/stride + 1 ;
  tempDimensions[2] = filterHeight*filterWidth*filterDepth*numGroups ;

  if (verbosity > 0) {
    double const MB = 1024.0*1024.0 ;
    mexPrintf("gconv: mode %s; %s\n", gpuMode?"gpu":"cpu", backMode?"backward":"forward") ;
    mexPrintf("gconv: stride: %d, pad: %d, numGroups: %d\n", stride, pad, numGroups) ;
    mexPrintf("gconv: data: %d x %d x %d x %d [%.1f MB]\n",
              height, width, depth, numImages,
              (double)(height*width*depth*numImages*4)/MB) ;
    mexPrintf("gconv: filters: %d x %d x %d x %d [%.1f MB]\n",
              filterHeight, filterWidth, filterDepth, numFilters,
              (double)(filterHeight*filterWidth*filterDepth*numFilters*4)/MB) ;
    mexPrintf("gconv: result: %d x %d x %d x %d [%.1f MB]\n",
              resultDimensions[0], resultDimensions[1], resultDimensions[2], resultDimensions[3],
              (double)(resultDimensions[0]*resultDimensions[1]*resultDimensions[2]*resultDimensions[3]*4)/MB) ;
    if (backMode) {
      mexPrintf("gconv: der: %d x %d x %d x %d [%.1f MB]\n",
                derHeight, derWidth, derDepth, numDerImages,
                (double)(derHeight*derWidth*derDepth*numDerImages*4)/MB) ;
      mexPrintf("gconv: dfilters: %d x %d x %d x %d [%.1f MB]\n",
                dfiltersDimensions[0], dfiltersDimensions[1], dfiltersDimensions[2], dfiltersDimensions[3],
                (double)(dfiltersDimensions[0]*dfiltersDimensions[1]*dfiltersDimensions[2]*dfiltersDimensions[3]*4)/MB) ;
    }
    mexPrintf("gconv: temp: %d x %d x %d [%.1f MB]\n",
              tempDimensions[0], tempDimensions[1], tempDimensions[2],
              (double)(tempDimensions[0]*tempDimensions[1]*tempDimensions[2]*4)/MB) ;
  }

  if (backMode) {
    if (derHeight != tempDimensions[0] ||
        derWidth != tempDimensions[1] ||
        derDepth != numFilters ||
        numDerImages != numImages)
    {
      mexErrMsgTxt("DER dimensions are incompatible with X and FILTERS.") ;
    }
  }

  if (numGroups * filterDepth != depth) {
    mexErrMsgTxt("The filter depth does not divide the image depth.") ;
  }

  if (numFilters % numGroups != 0) {
    mexErrMsgTxt("The number of filter groups does not divide the total number of filters.") ;
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

  if (gpuMode) {
#ifdef ENABLE_GPU
    tempGpu = mxGPUCreateGPUArray(3, tempDimensions,
                                  mxSINGLE_CLASS,
                                  mxREAL,
                                  MX_GPU_DO_NOT_INITIALIZE) ;
    if (!backMode || nout > 1) {
      resultGpu = mxGPUCreateGPUArray(4, resultDimensions,
                                      mxSINGLE_CLASS,
                                      mxREAL,
                                      MX_GPU_DO_NOT_INITIALIZE) ;
    }
    if (backMode) {
      /* note that this buffer must be initialized to zero */
      dfiltersGpu = mxGPUCreateGPUArray(4, dfiltersDimensions,
                                        mxSINGLE_CLASS,
                                        mxREAL,
                                        MX_GPU_INITIALIZE_VALUES) ;
    }
#else
    assert(false) ;
#endif
  } else {
    tempArray = mxCreateNumericArray(3, tempDimensions,
                                     mxSINGLE_CLASS,
                                     mxREAL) ;
    if (!backMode || nout > 1) {
      resultArray = mxCreateNumericArray(4, resultDimensions,
                                         mxSINGLE_CLASS,
                                         mxREAL) ;
    }
    if (backMode) {
      dfiltersArray = mxCreateNumericArray(4, dfiltersDimensions,
                                           mxSINGLE_CLASS,
                                           mxREAL);
    }
  }

  for (int image = 0 ; image < numImages ; ++image) {
    /*
     temp (phi(x)): m x k
     filters, dfilters: k x n (for one group of filters)
     der (dzdy) : m x n (for one group of filters)
     res (y) : m x n (for one group of filters)
     */
    ptrdiff_t dataImOffset = (width*height*depth) * image ;
    ptrdiff_t resImOffset = (resultDimensions[0]*resultDimensions[1]*resultDimensions[2]) * image ;
    ptrdiff_t derImOffset = (tempDimensions[0]*tempDimensions[1]*numFilters) * image ;
    ptrdiff_t m = tempDimensions[0]*tempDimensions[1] ; /* num output pixels */
    ptrdiff_t n = numFilters/numGroups ; /* num filters per group */
    ptrdiff_t k = filterHeight*filterWidth*filterDepth ; /* filter volume */
    char OP_N = 'n' ;
    char OP_T = 't' ;

    if (backMode) {
      /* ---------------------------------------------------------- */
      /*                                              Backward mode */
      /* ---------------------------------------------------------- */

      /* derivative w.r.t. filters dz/dF */
      if (gpuMode) {
#ifdef ENABLE_GPU
        im2col_gpu<float>((float const*)mxGPUGetDataReadOnly(dataGpu) + dataImOffset,
                          depth, width, height,
                          filterHeight,
                          stride, pad,
                          (float *)mxGPUGetData(tempGpu)) ;
#else
        assert(false) ;
#endif
      } else {
        im2col_cpu<float>((float const*)mxGetData(in[IN_DATA]) + dataImOffset,
                          depth, width, height,
                          filterHeight,
                          stride, pad,
                          (float *)mxGetData(tempArray)) ;
      }
      for (int g = 0 ; g < numGroups ; ++ g) {
        ptrdiff_t filterOffset = k * n * g ;
        ptrdiff_t tempOffset = m * k * g ;
        ptrdiff_t derGroupOffset = m * n * g ;
        float alpha = 1 ;
        float beta = 1 ;
        if (gpuMode) {
#ifdef ENABLE_GPU
          cublasSgemm(handle,
                      CUBLAS_OP_T, CUBLAS_OP_N,
                      (int)k, (int)n, (int)m,
                      &alpha,
                      (float const*)mxGPUGetDataReadOnly(tempGpu) + tempOffset, (int)m,
                      (float const*)mxGPUGetDataReadOnly(derGpu) + derImOffset + derGroupOffset, (int)m,
                      &beta,
                      (float*)mxGPUGetData(dfiltersGpu) + filterOffset, (int)k) ;
#else
          assert(false) ;
#endif
        } else {
          sgemm(&OP_T, &OP_N,
                &k, &n, &m,
                &alpha,
                (float*)mxGetData(tempArray) + tempOffset, &m,
                (float*)mxGetData(in[IN_DER]) + derImOffset + derGroupOffset, &m,
                &beta,
                (float*)mxGetData(dfiltersArray) + filterOffset, &k) ;
        }
      }

      /* derivative w.r.t. input image dz/dX */
      if (nout > 1) {
        for (int g = 0 ; g < numGroups ; ++ g) {
          ptrdiff_t filterOffset = k * n * g ;
          ptrdiff_t tempOffset = m * k * g ;
          ptrdiff_t derGroupOffset = m * n * g ;
          float alpha = 1 ;
          float beta = 0 ;
          if (gpuMode) {
#ifdef ENABLE_GPU
            cublasSgemm(handle,
                        CUBLAS_OP_N, CUBLAS_OP_T,
                        (int)m, (int)k, (int)n,
                        &alpha,
                        (float const*)mxGPUGetDataReadOnly(derGpu) + derImOffset + derGroupOffset, (int)m,
                        (float const*)mxGPUGetDataReadOnly(filtersGpu) + filterOffset, (int)k,
                        &beta,
                        (float*)mxGPUGetData(tempGpu) + tempOffset, (int)m) ;
#else
            assert(false) ;
#endif
          } else {
            sgemm(&OP_N, &OP_T,
                  &m, &k, &n,
                  &alpha,
                  (float*)mxGetData(in[IN_DER]) + derImOffset + derGroupOffset, &m,
                  (float*)mxGetData(in[IN_FILTERS]) + filterOffset, &k,
                  &beta,
                  (float*)mxGetData(tempArray) + tempOffset, &m) ;
          }
        }
        if (gpuMode) {
#ifdef ENABLE_GPU
          col2im_gpu<float>((float*)mxGPUGetData(tempGpu),
                            depth, width, height,
                            filterHeight,
                            stride, pad,
                            (float*)mxGPUGetData(resultGpu) + resImOffset) ;
#else
          assert(false) ;
#endif
        } else {
          col2im_cpu<float>((float*)mxGetData(tempArray),
                            depth, width, height,
                            filterHeight,
                            stride, pad,
                            (float*)mxGetData(resultArray) + resImOffset) ;
        }
      }
    } else {
      /* ---------------------------------------------------------- */
      /*                                               Forward mode */
      /* ---------------------------------------------------------- */
      if (gpuMode) {
#ifdef ENABLE_GPU
        im2col_gpu<float>((float const*)mxGPUGetDataReadOnly(dataGpu) + dataImOffset,
                          depth, width, height,
                          filterHeight,
                          stride, pad,
                          (float *)mxGPUGetData(tempGpu)) ;
#else
        assert(false) ;
#endif
      } else {
        im2col_cpu<float>((float const*)mxGetData(in[IN_DATA]) + dataImOffset,
                          depth, width, height,
                          filterHeight,
                          stride, pad,
                          (float *)mxGetData(tempArray)) ;
      }
      for (int g = 0 ; g < numGroups ; ++ g) {
        ptrdiff_t filterOffset = k * n * g ;
        ptrdiff_t tempOffset = m * k * g ;
        ptrdiff_t resultGroupOffset = m * n * g  ;
        float alpha = 1 ;
        float beta = 0 ;
        if (gpuMode) {
#ifdef ENABLE_GPU
          cublasSgemm(handle,
                      CUBLAS_OP_N, CUBLAS_OP_N,
                      (int)m, (int)n, (int)k,
                      &alpha,
                      (float const*)mxGPUGetDataReadOnly(tempGpu) + tempOffset, (int)m,
                      (float const*)mxGPUGetDataReadOnly(filtersGpu) + filterOffset, (int)k,
                      &beta,
                      (float*)mxGPUGetData(resultGpu) + resImOffset + resultGroupOffset, (int)m) ;
#else
          assert(false) ;
#endif
        } else {
          sgemm(&OP_N, &OP_N,
                &m, &n, &k,
                &alpha,
                (float*)mxGetData(tempArray) + tempOffset, &m,
                (float*)mxGetData(in[IN_FILTERS]) + filterOffset, &k,
                &beta,
                (float*)mxGetData(resultArray) + resImOffset + resultGroupOffset, &m) ;
        }
      }
    }
  }


  /* -------------------------------------------------------------- */
  /*                                                        Cleanup */
  /* -------------------------------------------------------------- */
  if (gpuMode) {
#ifdef ENABLE_GPU
    if (backMode) {
      out[OUT_RESULT] = mxGPUCreateMxArrayOnGPU(dfiltersGpu) ;
      if (nout > 1) {
        out[OUT_RESULT2] = mxGPUCreateMxArrayOnGPU(resultGpu) ;
      }
    } else {
      out[OUT_RESULT] = mxGPUCreateMxArrayOnGPU(resultGpu) ;
    }
    mxGPUDestroyGPUArray(dataGpu) ;
    mxGPUDestroyGPUArray(filtersGpu) ;
    if (!backMode || nout > 1) { mxGPUDestroyGPUArray(resultGpu) ; }
    if (backMode) { mxGPUDestroyGPUArray(dfiltersGpu) ; }
    mxGPUDestroyGPUArray(tempGpu) ;
    cublasDestroy(handle);
#else
    assert(false) ;
#endif
  } else {
    mxDestroyArray(tempArray);
    if (backMode) {
      out[OUT_RESULT] = dfiltersArray ;
      if (nout > 1) { out[OUT_RESULT2] = resultArray ; }
    } else {
      out[OUT_RESULT] = resultArray ;
    }
  }
}
