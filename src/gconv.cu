/* rip-off convolution from decaf and port it to MATLAB and gpuArrays */

#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cublas_v2.h>
#include <blas.h>
#include <iostream>

#include "bits/mexutils.h"
#include "bits/im2col.cpp"

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


inline vl_uindex divup(vl_uindex i, vl_uindex d)
{
  return (i + d - 1) / d ;
}

void mexFunction(int nout, mxArray *out[],
                 int nin, mxArray const *in[])
{
  mxClassID dataClassID ;
  mxClassID filtersClassID ;
  mxClassID derClassID ;
  mxGPUArray const *dataGpu ;
  mxGPUArray const *filtersGpu ;
  mxGPUArray const *derGpu ;

  mxGPUArray *resultGpu ;
  mxGPUArray *dfiltersGpu ;
  mxGPUArray *tempGpu ;

  mxArray *resultArray ;
  mxArray *dfiltersArray ;
  mxArray *tempArray ;

  cublasStatus_t stat;
  cublasHandle_t handle;

  size_t height, width, depth, numImages ;
  size_t filterHeight, filterWidth, filterDepth, numFilters ;
  size_t derHeight, derWidth, derDepth, numDerImages ;
  int stride = 1 ;
  int pad = 0 ;
  mwSize dataNumDimensions ;
  mwSize filtersNumDimensions ;
  mwSize derNumDimensions ;
  mwSize const * dataDimensions ;
  mwSize const * filtersDimensions ;
  mwSize const * derDimensions ;
  mwSize resultDimensions [4] ;
  mwSize dfiltersDimensions [4] ;
  mwSize tempDimensions [3] ;

  bool gpuMode = false ;
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
    if (backMode) {
      if (!mxIsGPUArray(in[IN_DER])) {
        mexErrMsgTxt("DATA is a GPU array but FILTERS is not.") ;
      }
      derGpu = mxGPUCreateFromMxArray(in[IN_DER]) ;
      derClassID = mxGPUGetClassID(derGpu) ;
      derNumDimensions = mxGPUGetNumberOfDimensions(derGpu) ;
      derDimensions = mxGPUGetDimensions(derGpu) ;
    }
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
  tempDimensions[2] = filterHeight*filterWidth*filterDepth ;

  if (verbosity > 0) {
    double const MB = 1024.0*1024.0 ;
    mexPrintf("gconv: mode %s; %s\n", gpuMode?"gpu":"cpu", backMode?"backward":"forward") ;
    mexPrintf("gconf: stride: %d, pad: %d\n", stride, pad) ;
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
    if (backMode) {
      /* ---------------------------------------------------------- */
      /*                                              Backward mode */
      /* ---------------------------------------------------------- */
      ptrdiff_t dataOffset = (width*height*depth) * image ;
      ptrdiff_t derOffset = (tempDimensions[0]*tempDimensions[1]*numFilters) * image ;
      ptrdiff_t resultOffset = (resultDimensions[0]*resultDimensions[1]*resultDimensions[2]) * image ;
      {
        float alpha = 1 ;
        float beta = 1 ;
        char opA = 't' ;
        char opB = 'n' ;
        ptrdiff_t m = tempDimensions[2] ; /* = filter volume */
        ptrdiff_t n = numFilters ;
        ptrdiff_t k = tempDimensions[0]*tempDimensions[1] ;
        /* derivative w.r.t. filters dz/dF */
        if (gpuMode) {
          im2col_gpu<float>((float const*)mxGPUGetDataReadOnly(dataGpu) + dataOffset,
                            depth, width, height,
                            filterHeight,
                            stride, pad,
                            (float *)mxGPUGetData(tempGpu)) ;
          cublasSgemm(handle,
                      (opA == 'n') ? CUBLAS_OP_N : CUBLAS_OP_T,
                      (opB == 'n') ? CUBLAS_OP_N : CUBLAS_OP_T,
                      (int)m, (int)n, (int)k,
                      &alpha,
                      (float const*)mxGPUGetDataReadOnly(tempGpu), (opA == 'n') ? (int)m : (int)k,
                      (float const*)mxGPUGetDataReadOnly(derGpu) + derOffset, (opB == 'n') ? (int)k : (int)n,
                      &beta,
                      (float*)mxGPUGetData(dfiltersGpu), (int)m) ;
        } else {
#if 1
          im2col_cpu<float>((float const*)mxGetData(in[IN_DATA]) + dataOffset,
                            depth, width, height,
                            filterHeight,
                            stride, pad,
                            (float *)mxGetData(tempArray)) ;
          sgemm(&opA, &opB,
                &m, &n, &k,
                &alpha,
                (float*)mxGetData(tempArray), (opA == 'n') ? &m : &k,
                (float*)mxGetData(in[IN_DER]) + derOffset,(opB == 'n') ? &k : &n,
                &beta,
                (float*)mxGetData(dfiltersArray), &m) ;
#endif
        }
      }
      /* derivative w.r.t. input image dz/dX */
      if (nout > 1) {
        float alpha = 1 ;
        float beta = 0 ;
        char opA = 'n' ;
        char opB = 't' ;
        ptrdiff_t m = tempDimensions[0]*tempDimensions[1] ;
        ptrdiff_t n = tempDimensions[2] ;
        ptrdiff_t k = numFilters ;
        if (gpuMode) {
          cublasSgemm(handle,
                      (opA == 'n') ? CUBLAS_OP_N : CUBLAS_OP_T,
                      (opB == 'n') ? CUBLAS_OP_N : CUBLAS_OP_T,
                      (int)m, (int)n, (int)k,
                      &alpha,
                      (float const*)mxGPUGetDataReadOnly(derGpu) + derOffset, (opA == 'n') ? (int)m : (int)k,
                      (float const*)mxGPUGetDataReadOnly(filtersGpu), (opB == 'n') ? (int)k : (int)n,
                      &beta,
                      (float*)mxGPUGetData(tempGpu), (int)m) ;
          col2im_gpu<float>((float*)mxGPUGetData(tempGpu),
                            depth, width, height,
                            filterHeight,
                            stride, pad,
                            (float*)mxGPUGetData(resultGpu) + resultOffset) ;
        } else {
          // overwrite temp
          sgemm(&opA, &opB,
                &m, &n, &k,
                &alpha,
                (float*)mxGetData(in[IN_DER]) + derOffset, (opA == 'n') ? &m : &k,
                (float*)mxGetData(in[IN_FILTERS]),(opB == 'n') ? &k : &n,
                &beta,
                (float*)mxGetData(tempArray), &m) ;
#if 1

          col2im_cpu<float>((float*)mxGetData(tempArray),
                            depth, width, height,
                            filterHeight,
                            stride, pad,
                            (float*)mxGetData(resultArray) + resultOffset) ;
#endif

        }
      }
    } else {
      /* ---------------------------------------------------------- */
      /*                                               Forward mode */
      /* ---------------------------------------------------------- */
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
        im2col_gpu<float>((float const*)mxGPUGetDataReadOnly(dataGpu) + dataOffset,
                          depth, width, height,
                          filterHeight,
                          stride, pad,
                          (float *)mxGPUGetData(tempGpu)) ;
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

      } else {
#if 1
        if (opA == 't') {
#if 0
          im2row_cpu<float>((float const*)mxGetData(in[IN_DATA]) + dataOffset,
                            depth, width, height,
                            filterHeight,
                            stride, pad,
                            (float *)mxGetData(tempArray)) ;
#endif
        } else {
          im2col_cpu<float>((float const*)mxGetData(in[IN_DATA]) + dataOffset,
                            depth, width, height,
                            filterHeight,
                            stride, pad,
                            (float *)mxGetData(tempArray)) ;
        }
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
  }


  /* -------------------------------------------------------------- */
  /*                                                        Cleanup */
  /* -------------------------------------------------------------- */
  if (gpuMode) {
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
