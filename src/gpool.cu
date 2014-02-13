#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cublas_v2.h>
#include <blas.h>
#include <iostream>

#include "bits/pooling.cpp"

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
  mxGPUArray const *dataGpu ;
  mxGPUArray const *derGpu ;
  mxGPUArray *resultGpu ;
  mxArray *resultArray ;

  size_t height, width, depth, numImages ;
  size_t poolHeight, poolWidth, poolStride ;
  size_t derHeight, derWidth, derDepth, numDerImages ;

  mwSize dataNumDimensions ;
  mwSize derNumDimensions ;
  mwSize const * dataDimensions ;
  mwSize const * derDimensions ;
  mwSize resultDimensions [4] ;
  mwSize tempDimensions [3] ;

  bool gpuMode = false ;
  bool backMode = false ;
  int verbosiy = 1 ;

  /* -------------------------------------------------------------- */
  /*                                            Check the arguments */
  /* -------------------------------------------------------------- */

  /* Throw an error if the input is not a GPU array. */
  if (nin != 2 && nin != 3) {
    mexErrMsgTxt("The arguments are neither two or three.") ;
  }

  backMode = (nin == 3) ;
  gpuMode = mxIsGPUArray(in[IN_DATA]) ;

  if (!mxIsNumeric(in[IN_SIZE]) ||
       mxGetClassID(in[IN_SIZE]) != mxDOUBLE_CLASS ||
       mxIsComplex(in[IN_SIZE]) ||
       mxGetNumberOfDimensions(in[IN_SIZE]) != 2)
  {
    mexErrMsgTxt("SIZE is not a plain 2 vector.") ;
  }
  poolStride = 1 ;
  poolHeight = mxGetPr(in[IN_SIZE])[0] ;
  poolWidth = mxGetPr(in[IN_SIZE])[1] ;


  if (gpuMode) {
    mxInitGPU() ;
    dataGpu = mxGPUCreateFromMxArray(in[IN_DATA]) ;
    dataClassID = mxGPUGetClassID(dataGpu) ;
    dataNumDimensions = mxGPUGetNumberOfDimensions(dataGpu) ;
    dataDimensions = mxGPUGetDimensions(dataGpu) ;
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

  if (poolWidth != poolHeight) {
    mexErrMsgTxt("Non-square FILTERS not supported yet.") ;
  }

  if (!backMode) {
    resultDimensions[0] = height - poolHeight + 1 ;
    resultDimensions[1] = width - poolWidth + 1 ;
    resultDimensions[2] = depth ;
    resultDimensions[3] = numImages ;
  } else {
    resultDimensions[0] = height ;
    resultDimensions[1] = width ;
    resultDimensions[2] = depth ;
    resultDimensions[3] = numImages ;
  }

  tempDimensions[0] = height - poolHeight + 1 ;
  tempDimensions[1] = width - poolWidth + 1 ;
  tempDimensions[2] = poolHeight*poolWidth*depth ;

  if (verbosiy > 0) {
    double const MB = 1024.0*1024.0 ;
    mexPrintf("gconv: mode %s; %s\n", gpuMode?"gpu":"cpu", backMode?"backward":"forward") ;
    mexPrintf("gconv: data: %d x %d x %d x %d [%.1f MB]\n",
              height, width, depth, numImages,
              (double)(height*width*depth*numImages*4)/MB) ;
    mexPrintf("gconv: pooling: %d x %d\n", poolHeight, poolWidth);
    mexPrintf("gconv: result: %d x %d x %d x %d [%.1f MB]\n",
              resultDimensions[0], resultDimensions[1], resultDimensions[2], resultDimensions[3],
              (double)(resultDimensions[0]*resultDimensions[1]*resultDimensions[2]*resultDimensions[3]*4)/MB) ;
    if (backMode) {
      mexPrintf("gconv: der: %d x %d x %d x %d [%.1f MB]\n",
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

  /* -------------------------------------------------------------- */
  /*                                                    Do the work */
  /* -------------------------------------------------------------- */
  // im2col should be called im2row

  if (gpuMode) {
    resultGpu = mxGPUCreateGPUArray(4, resultDimensions,
                                    mxSINGLE_CLASS,
                                    mxREAL,
                                    MX_GPU_INITIALIZE_VALUES) ;
//                                    MX_GPU_DO_NOT_INITIALIZE) ;
  } else {
    resultArray = mxCreateNumericArray(4, resultDimensions,
                                       mxSINGLE_CLASS,
                                       mxREAL) ;
  }

  for (int image = 0 ; image < numImages ; ++image) {
    ptrdiff_t dataOffset = (width*height*depth) * image ;
    ptrdiff_t derOffset = (derDimensions[0]*derDimensions[1]*derDimensions[2]) * image ;
    ptrdiff_t resultOffset = (resultDimensions[0]*resultDimensions[1]*resultDimensions[2]) * image ;

    if (backMode) {
      /* ---------------------------------------------------------- */
      /*                                              Backward mode */
      /* ---------------------------------------------------------- */
      if (gpuMode) {
        maxPoolingBackward_gpu<float>((float*)mxGetData(resultArray) + resultOffset,
                                      (float const*)mxGetData(in[IN_DATA]) + dataOffset,
                                      (float const*)mxGetData(in[IN_DER]) + derOffset,
                                      height, width, depth,
                                      poolWidth, poolStride) ;
      } else {
        maxPoolingBackward_cpu<float>((float*)mxGetData(resultArray) + resultOffset,
                                      (float const*)mxGetData(in[IN_DATA]) + dataOffset,
                                      (float const*)mxGetData(in[IN_DER]) + derOffset,
                                      height, width, depth,
                                      poolWidth, poolStride) ;
      }
    } else {
      /* ---------------------------------------------------------- */
      /*                                               Forward mode */
      /* ---------------------------------------------------------- */
      if (gpuMode) {
        maxPooling_gpu<float>((float*)mxGPUGetData(resultGpu) + resultOffset,
                              (float const*)mxGPUGetDataReadOnly(dataGpu) + dataOffset,
                              height, width, depth,
                              poolWidth, poolStride) ;
      } else {
        maxPooling_cpu<float>((float*)mxGetData(resultArray) + resultOffset,
                              (float const*)mxGetData(in[IN_DATA]) + dataOffset,
                              height, width, depth,
                              poolWidth, poolStride) ;
      }
    }
  }

  /* -------------------------------------------------------------- */
  /*                                                        Cleanup */
  /* -------------------------------------------------------------- */
  if (gpuMode) {
    out[OUT_RESULT] = mxGPUCreateMxArrayOnGPU(resultGpu) ;
    mxGPUDestroyGPUArray(dataGpu) ;
    mxGPUDestroyGPUArray(resultGpu) ;
  } else {
    out[OUT_RESULT] = resultArray ;
  }
}
