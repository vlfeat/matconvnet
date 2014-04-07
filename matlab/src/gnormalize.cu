/** @file gnormalize.cu
 ** @brief Normalization block
 ** @author Andrea Vedaldi
 **/

#include "mex.h"
#ifdef ENABLE_GPU
#include "gpu/mxGPUArray.h"
#endif
#include "bits/mexutils.h"
#include "bits/normalize.hpp"
#include <assert.h>

/* option codes */
enum {
  opt_verbose = 0
} ;

/* options */
vlmxOption  options [] = {
  {"Verbose",          0,   opt_verbose           },
  {0,                  0,   0                     }
} ;

enum {
  IN_DATA = 0, IN_PARAM, IN_DER, IN_END
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
  size_t derHeight, derWidth, derDepth, numDerImages ;
  size_t normDepth ;
  double normAlpha ;
  double normKappa ;
  double normBeta ;

  mwSize dataNumDimensions ;
  mwSize derNumDimensions ;
  mwSize const * dataDimensions ;
  mwSize const * derDimensions ;
  mwSize resultDimensions [4] ;

#if ENABLE_GPU
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

  if (!mxIsNumeric(in[IN_PARAM]) ||
       mxGetClassID(in[IN_PARAM]) != mxDOUBLE_CLASS ||
       mxIsComplex(in[IN_PARAM]) ||
       mxGetNumberOfElements(in[IN_PARAM]) != 4)
  {
    mexErrMsgTxt("PARAM is not a plain 4 vector.") ;
  }
  normDepth = (size_t) mxGetPr(in[IN_PARAM])[0]  ;
  normKappa = mxGetPr(in[IN_PARAM])[1]  ;
  normAlpha = mxGetPr(in[IN_PARAM])[2]  ;
  normBeta = mxGetPr(in[IN_PARAM])[3]  ;

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

  resultDimensions[0] = height ;
  resultDimensions[1] = width ;
  resultDimensions[2] = depth ;
  resultDimensions[3] = numImages ;

  if (verbosity > 0) {
    double const MB = 1024.0*1024.0 ;
    mexPrintf("gnormalize: mode %s; %s\n", gpuMode?"gpu":"cpu", backMode?"backward":"forward") ;
    mexPrintf("gnormalize: data: %d x %d x %d x %d [%.1f MB]\n",
              height, width, depth, numImages,
              (double)(height*width*depth*numImages*4)/MB) ;
    mexPrintf("gnormalize: (depth,kappa,alpha,beta): (%d,%g,%g,%g)\n",
              normDepth, normKappa, normAlpha, normBeta) ;
    mexPrintf("gnormalize: result: %d x %d x %d x %d [%.1f MB]\n",
              resultDimensions[0], resultDimensions[1], resultDimensions[2], resultDimensions[3],
              (double)(resultDimensions[0]*resultDimensions[1]*resultDimensions[2]*resultDimensions[3]*4)/MB) ;
    if (backMode) {
      mexPrintf("gnormalize: der: %d x %d x %d x %d [%.1f MB]\n",
                derHeight, derWidth, derDepth, numDerImages,
                (double)(derHeight*derWidth*derDepth*numDerImages*4)/MB) ;
    }
  }

  if (backMode) {
    if (derHeight != height ||
        derWidth != width ||
        derDepth != depth ||
        numDerImages != numImages)
    {
      mexErrMsgTxt("DER dimensions are incompatible with X.") ;
    }
  }

  if (normDepth < 1) {
    mexErrMsgTxt("The normalization depth is smaller than 1.") ;
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
        normalizeBackward_gpu<float>((float*)mxGPUGetData(resultGpu) + resultOffset,
                                     (float const*)mxGPUGetDataReadOnly(dataGpu) + dataOffset,
                                     (float const*)mxGPUGetDataReadOnly(derGpu) + derOffset,
                                     height, width, depth,
                                     normDepth, normKappa, normAlpha, normBeta) ;
#else
        assert(false) ;
#endif
      } else {
        normalizeBackward_cpu<float>((float*)mxGetData(resultArray) + resultOffset,
                                     (float const*)mxGetData(in[IN_DATA]) + dataOffset,
                                     (float const*)mxGetData(in[IN_DER]) + derOffset,
                                     height, width, depth,
                                     normDepth, normKappa, normAlpha, normBeta) ;
      }
    } else {
      /* ---------------------------------------------------------- */
      /*                                               Forward mode */
      /* ---------------------------------------------------------- */
      if (gpuMode) {
#ifdef ENABLE_GPU
        normalize_gpu<float>((float*)mxGPUGetData(resultGpu) + resultOffset,
                             (float const*)mxGPUGetDataReadOnly(dataGpu) + dataOffset,
                             height, width, depth,
                             normDepth, normKappa, normAlpha, normBeta) ;
#else
        assert(false) ;
#endif
      } else {
        normalize_cpu<float>((float*)mxGetData(resultArray) + resultOffset,
                             (float const*)mxGetData(in[IN_DATA]) + dataOffset,
                             height, width, depth,
                             normDepth, normKappa, normAlpha, normBeta) ;
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
