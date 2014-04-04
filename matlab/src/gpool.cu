/** @file gpool.cu
 ** @brief Max-pooling block
 ** @author Andrea Vedaldi
 **/

#include "mex.h"
#ifdef ENABLE_GPU
#include "gpu/mxGPUArray.h"
#endif
#include "bits/mexutils.h"
#include "bits/pooling.hpp"
#include <blas.h>
#include <assert.h>

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
  IN_DATA = 0, IN_SIZE, IN_DER, IN_END
} ;

enum {
  OUT_RESULT = 0, OUT_END
} ;

void mexFunction(int nout, mxArray *out[],
                 int nin, mxArray const *in[])
{
  mxClassID dataClassID ;
  mxClassID derClassID ;
  mxArray *resultArray ;

  size_t height, width, depth, numImages ;
  size_t poolHeight, poolWidth ;
  size_t derHeight, derWidth, derDepth, numDerImages ;
  int stride = 1 ;
  int pad = 0 ;

  mwSize dataNumDimensions ;
  mwSize derNumDimensions ;
  mwSize const * dataDimensions ;
  mwSize const * derDimensions ;
  mwSize resultDimensions [4] ;
  mwSize tempDimensions [3] ;

#ifdef ENABLE_GPU
  mxGPUArray const *dataGpu ;
  mxGPUArray const *derGpu ;
  mxGPUArray *resultGpu ;
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

  /* Throw an error if the input is not a GPU array. */
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

  if (!mxIsNumeric(in[IN_SIZE]) ||
       mxGetClassID(in[IN_SIZE]) != mxDOUBLE_CLASS ||
       mxIsComplex(in[IN_SIZE]) ||
       mxGetNumberOfElements(in[IN_SIZE]) != 2)
  {
    mexErrMsgTxt("SIZE is not a plain 2 vector.") ;
  }
  poolHeight = mxGetPr(in[IN_SIZE])[0] ;
  poolWidth = mxGetPr(in[IN_SIZE])[1] ;

  if (gpuMode) {
#ifdef ENABLE_GPU
    mxInitGPU() ;
    dataGpu = mxGPUCreateFromMxArray(in[IN_DATA]) ;
    dataClassID = mxGPUGetClassID(dataGpu) ;
    dataNumDimensions = mxGPUGetNumberOfDimensions(dataGpu) ;
    dataDimensions = mxGPUGetDimensions(dataGpu) ;
    if (backMode) {
      if (!mxIsGPUArray(in[IN_DER])) {
        mexErrMsgTxt("DATA is a GPU array but DER is not.") ;
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
    dataClassID = mxGetClassID(in[IN_DATA]) ;
    dataNumDimensions = mxGetNumberOfDimensions(in[IN_DATA]) ;
    dataDimensions = mxGetDimensions(in[IN_DATA]) ;
    if (backMode) {
      if (!mxIsNumeric(in[IN_DER])) {
        mexErrMsgTxt("DATA is a numeric array but DER is not.") ;
      }
      derClassID = mxGetClassID(in[IN_DER]) ;
      derNumDimensions = mxGetNumberOfDimensions(in[IN_DER]) ;
      derDimensions = mxGetDimensions(in[IN_DER]) ;
    }
  }

  if (dataClassID != mxSINGLE_CLASS) {
    mexErrMsgTxt("DATA is not of class SINGLE.");
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

  if (poolWidth != poolHeight) {
    mexErrMsgTxt("Non-square POOL not supported yet.") ;
  }

  if (!backMode) {
    resultDimensions[0] = (height + 2*pad - poolHeight)/stride + 1 ;
    resultDimensions[1] = (width + 2*pad - poolWidth)/stride + 1 ;
    resultDimensions[2] = depth ;
    resultDimensions[3] = numImages ;
  } else {
    resultDimensions[0] = height ;
    resultDimensions[1] = width ;
    resultDimensions[2] = depth ;
    resultDimensions[3] = numImages ;
  }

  tempDimensions[0] = (height + 2*pad - poolHeight)/stride + 1 ;
  tempDimensions[1] = (width + 2*pad - poolHeight)/stride + 1 ;
  tempDimensions[2] = poolHeight*poolWidth*depth ;

  if (verbosity > 0) {
    double const MB = 1024.0*1024.0 ;
    mexPrintf("gpool: mode %s; %s\n", gpuMode?"gpu":"cpu", backMode?"backward":"forward") ;
    mexPrintf("gpool: stride: %d, pad: %d\n", stride, pad) ;
    mexPrintf("gpool: data: %d x %d x %d x %d [%.1f MB]\n",
              height, width, depth, numImages,
              (double)(height*width*depth*numImages*4)/MB) ;
    mexPrintf("gpool: pooling: %d x %d\n", poolHeight, poolWidth);
    mexPrintf("gpool: result: %d x %d x %d x %d [%.1f MB]\n",
              resultDimensions[0], resultDimensions[1], resultDimensions[2], resultDimensions[3],
              (double)(resultDimensions[0]*resultDimensions[1]*resultDimensions[2]*resultDimensions[3]*4)/MB) ;
    if (backMode) {
      mexPrintf("gpool: der: %d x %d x %d x %d [%.1f MB]\n",
                derHeight, derWidth, derDepth, numDerImages,
                (double)(derHeight*derWidth*derDepth*numDerImages*4)/MB) ;
    }
  }

  if (backMode) {
    if (derHeight != tempDimensions[0] ||
        derWidth != tempDimensions[1] ||
        derDepth != depth ||
        numDerImages != numImages)
    {
      mexErrMsgTxt("DER dimensions are incompatible with X and pooling SIZE.") ;
    }
  }

  if (poolHeight != poolWidth) {
    mexErrMsgTxt("Non-square pooling not supported yet.");
  }

  if (height < poolHeight ||  width < poolWidth) {
    mexErrMsgTxt("Pooling SIZE is larger than the DATA.") ;
  }

  if (poolHeight == 0 || poolWidth == 0) {
    mexErrMsgTxt("A dimension of the pooling SIZE is void.") ;
  }

  if (pad >= poolWidth) {
    mexErrMsgTxt("PAD is larger or equal than the pooling size") ;
  }

  /* -------------------------------------------------------------- */
  /*                                                    Do the work */
  /* -------------------------------------------------------------- */
  // im2col should be called im2row

  if (gpuMode) {
#ifdef ENABLE_GPU
    resultGpu = mxGPUCreateGPUArray(4, resultDimensions,
                                    mxSINGLE_CLASS,
                                    mxREAL,
                                    MX_GPU_INITIALIZE_VALUES) ;
//                                    MX_GPU_DO_NOT_INITIALIZE) ;
#else
    assert(false) ;
#endif
  } else {
    resultArray = mxCreateNumericArray(4, resultDimensions,
                                       mxSINGLE_CLASS,
                                       mxREAL) ;
  }

  for (int image = 0 ; image < numImages ; ++image) {
    ptrdiff_t dataOffset = (width*height*depth) * image ;
    ptrdiff_t resultOffset = (resultDimensions[0]*resultDimensions[1]*resultDimensions[2]) * image ;

    if (backMode) {
      ptrdiff_t derOffset = (derDimensions[0]*derDimensions[1]*derDimensions[2]) * image ;

      /* ---------------------------------------------------------- */
      /*                                              Backward mode */
      /* ---------------------------------------------------------- */
      if (gpuMode) {
#ifdef ENABLE_GPU
        maxPoolingBackward_gpu<float>((float*)mxGPUGetData(resultGpu) + resultOffset,
                                      (float const*)mxGPUGetDataReadOnly(dataGpu) + dataOffset,
                                      (float const*)mxGPUGetDataReadOnly(derGpu) + derOffset,
                                      height, width, depth,
                                      poolWidth, stride, pad) ;
#else
        assert(false) ;
#endif
      } else {
        maxPoolingBackward_cpu<float>((float*)mxGetData(resultArray) + resultOffset,
                                      (float const*)mxGetData(in[IN_DATA]) + dataOffset,
                                      (float const*)mxGetData(in[IN_DER]) + derOffset,
                                      height, width, depth,
                                      poolWidth, stride, pad) ;
      }
    } else {
      /* ---------------------------------------------------------- */
      /*                                               Forward mode */
      /* ---------------------------------------------------------- */
      if (gpuMode) {
#ifdef ENABLE_GPU
        maxPooling_gpu<float>((float*)mxGPUGetData(resultGpu) + resultOffset,
                              (float const*)mxGPUGetDataReadOnly(dataGpu) + dataOffset,
                              height, width, depth,
                              poolWidth, stride, pad) ;
#else
        assert(false) ;
#endif
      } else {
        maxPooling_cpu<float>((float*)mxGetData(resultArray) + resultOffset,
                              (float const*)mxGetData(in[IN_DATA]) + dataOffset,
                              height, width, depth,
                              poolWidth, stride, pad) ;
      }
    }
  }

  /* -------------------------------------------------------------- */
  /*                                                        Cleanup */
  /* -------------------------------------------------------------- */
  if (gpuMode) {
#ifdef ENABLE_GPU
    out[OUT_RESULT] = mxGPUCreateMxArrayOnGPU(resultGpu) ;
    mxGPUDestroyGPUArray(dataGpu) ;
    mxGPUDestroyGPUArray(resultGpu) ;
#else
    assert(false) ;
#endif
  } else {
    out[OUT_RESULT] = resultArray ;
  }
}
