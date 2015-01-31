/** @file vl_nnpool.cu
 ** @brief Pooling block
 ** @author Andrea Vedaldi
 ** @author Karel Lenc
 **/

/*
Copyright (C) 2014 Andrea Vedaldi and Karel Lenc.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "bits/mexutils.h"
#include "bits/nnpooling.hpp"
#include "bits/datamex.hpp"

#if ENABLE_GPU
#include "bits/datacu.hpp"
#endif

#include <assert.h>

/* option codes */
enum {
  opt_stride = 0,
  opt_pad,
  opt_method,
  opt_verbose
} ;

/* options */
vlmxOption  options [] = {
  {"Stride",           1,   opt_stride            },
  {"Pad",              1,   opt_pad               },
  {"Method",           1,   opt_method            },
  {"Verbose",          0,   opt_verbose           },
  {0,                  0,   0                     }
} ;

/* ---------------------------------------------------------------- */
/*                                                          Context */
/* ---------------------------------------------------------------- */

vl::Context context ;

/*
 Resetting the context here resolves a crash when MATLAB quits and
 the ~Context function is implicitly called on unloading the MEX file.
 */
void atExit()
{
  context.reset() ;
}

/* ---------------------------------------------------------------- */
/*                                                       MEX driver */
/* ---------------------------------------------------------------- */

enum {
  IN_DATA = 0, IN_SIZE, IN_DEROUTPUT, IN_END
} ;

enum {
  OUT_RESULT = 0, OUT_END
} ;

void mexFunction(int nout, mxArray *out[],
                 int nin, mxArray const *in[])
{
  int poolWidth ;
  int poolHeight ;
  int strideX = 1 ;
  int strideY = 1 ;
  int padLeft = 0 ;
  int padRight = 0 ;
  int padTop = 0 ;
  int padBottom = 0 ;
  vl::PoolingMethod method = vl::MAX ;
  bool backMode = false ;

  int verbosity = 0 ;
  int opt ;
  int next = IN_END ;
  mxArray const *optarg ;

  /* -------------------------------------------------------------- */
  /*                                            Check the arguments */
  /* -------------------------------------------------------------- */

  mexAtExit(atExit) ;

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
        if (!vlmxIsPlainMatrix(optarg,-1,-1)) {
          mexErrMsgTxt("STRIDE is not a plain matrix.") ;
        }
        switch (mxGetNumberOfElements(optarg)) {
          case 1:
            strideY = (int)mxGetPr(optarg)[0] ;
            strideX = strideY ;
            break ;
          case 2:
            strideY = (int)mxGetPr(optarg)[0] ;
            strideX = (int)mxGetPr(optarg)[1] ;
            break ;
          default:
            mexErrMsgTxt("STRIDE has neither one nor two elements.") ;
        }
        break ;

      case opt_pad :
        if (!vlmxIsPlainMatrix(optarg,-1,-1)) {
          mexErrMsgTxt("PAD is not a plain matrix.") ;
        }
        switch (mxGetNumberOfElements(optarg)) {
          case 1:
            padLeft = (int)mxGetPr(optarg)[0] ;
            padRight = padLeft ;
            padTop = padLeft ;
            padBottom = padLeft ;
            break ;
          case 4:
            padTop = (int)mxGetPr(optarg)[0] ;
            padBottom = (int)mxGetPr(optarg)[1] ;
            padLeft = (int)mxGetPr(optarg)[2] ;
            padRight = (int)mxGetPr(optarg)[3] ;
            break ;
          default:
            mexErrMsgTxt("PAD has neither one nor four elements.") ;
        }
        break;

      case opt_method :
        if (!vlmxIsString(optarg,-1)) {
           vlmxError(vlmxErrInvalidArgument, "METHOD is not a string.") ;
        }
        if (vlmxIsEqualToStringI(optarg, "max")) {
          method = vl::MAX ;
        } else if (vlmxIsEqualToStringI(optarg, "avg")) {
          method = vl::AVERAGE ;
        } else {
          vlmxError(vlmxErrInvalidArgument, "METHOD is not a supported method.") ;
        }
        break;

      default:
        break ;
    }
  }

  vl::MexTensor data(in[IN_DATA]) ;
  vl::MexTensor derOutput(backMode ? in[IN_DEROUTPUT] : NULL) ;

  if (backMode && ! vl::areCompatible(data, derOutput)) {
    mexErrMsgTxt("DATA and DEROUTPUT are not both CPU or GPU arrays.") ;
  }

  if (!vlmxIsPlainMatrix(in[IN_SIZE],-1,-1)) {
    mexErrMsgTxt("SIZE is not a plain matrix.") ;
  }
  switch (mxGetNumberOfElements(in[IN_SIZE])) {
    case 1:
      poolHeight = mxGetPr(in[IN_SIZE])[0] ;
      poolWidth = poolHeight ;
      break ;
    case 2:
      poolHeight = mxGetPr(in[IN_SIZE])[0] ;
      poolWidth = mxGetPr(in[IN_SIZE])[1] ;
      break ;
    default:
      mexErrMsgTxt("SIZE has neither one nor two elements.") ;
  }

  /* Basic compatibility of geometry */
  if (strideX < 1 || strideY < 1) {
    mexErrMsgTxt("At least one element of STRIDE is smaller than one.") ;
  }
  if (poolHeight == 0 || poolWidth == 0) {
    mexErrMsgTxt("A dimension of the pooling SIZE is void.") ;
  }
  if (data.getHeight() + (padTop+padBottom) < poolHeight ||
      data.getWidth() + (padLeft+padRight) < poolWidth) {
    mexErrMsgTxt("The pooling window is larger than the DATA (including padding).") ;
  }
  if (padLeft < 0 ||
      padRight < 0 ||
      padTop < 0 ||
      padBottom < 0) {
    mexErrMsgTxt("An element of PAD is negative.") ;
  }
  if (padLeft >= poolWidth ||
      padRight >= poolWidth ||
      padTop >= poolHeight  ||
      padBottom >= poolHeight) {
    mexErrMsgTxt("A padding value is larger or equal than the size of the pooling window.") ;
  }

  /* Get the output geometry */
  vl::TensorGeometry outputGeom((data.getHeight() + (padTop+padBottom) - poolHeight)/strideY + 1,
                                (data.getWidth()  + (padLeft+padRight) - poolWidth)/strideX + 1,
                                data.getDepth(),
                                data.getSize()) ;

  if (backMode && (derOutput != outputGeom)) {
    mexErrMsgTxt("DEROUTPUT dimensions are incompatible with X and POOL.") ;
  }

  /* Create output buffers */
  vl::Device type = data.getMemoryType() ;
  vl::MexTensor output ;
  vl::MexTensor derData ;
  vl::MexTensor derFilters ;
  vl::MexTensor derBiases ;

  if (!backMode) {
    output = vl::MexTensor(type, outputGeom, 0) ;
  } else {
    derData = vl::MexTensor(type, data.getGeometry(), 0) ;
  }

  if (verbosity > 0) {
    mexPrintf("vl_nnpool: mode %s; %s\n", (data.getMemoryType()==vl::GPU)?"gpu":"cpu", backMode?"backward":"forward") ;
#if ENABLE_CUDNN
    if (data.getMemoryType()==vl::GPU) {
      mexPrintf("vl_nnpool: GPU algorithms: %s\n", context.getCudaHelper().isCudnnActive()?"cuDNN":"cuBLAS") ;
    }
#endif
    mexPrintf("vl_nnpool: stride: [%d %d], pad: [%d %d %d %d]\n",
              strideY, strideX,
              padTop, padBottom, padLeft, padRight) ;
    vl::print("vl_nnconv: data: ", data) ;
    mexPrintf("vl_nnpool: pooling: %d x %d\n", poolHeight, poolWidth);
    mexPrintf("vl_nnpool: method: %s\n", (method == vl::MAX) ? "max" : "avg") ;
    if (backMode) {
      vl::print("vl_nnconv: derOutput: ", derOutput) ;
      vl::print("vl_nnconv: derData: ", derData) ;
    } else {
      vl::print("vl_nnconv: output: ", output) ;
    }
  }

  /* -------------------------------------------------------------- */
  /*                                                    Do the work */
  /* -------------------------------------------------------------- */

  if (!backMode) {
    vl::nnpooling_forward(context,
                          output, data,
                          method,
                          poolHeight, poolWidth,
                          strideY, strideX,
                          padTop, padBottom, padLeft, padRight) ;
  } else {
    vl::nnpooling_backward(context,
                           derData, data, derOutput,
                           method,
                           poolHeight, poolWidth,
                           strideY, strideX,
                           padTop, padBottom, padLeft, padRight) ;
  }

  /* -------------------------------------------------------------- */
  /*                                                         Finish */
  /* -------------------------------------------------------------- */

  if (backMode) {
    out[OUT_RESULT] = derData.relinquish() ;
  } else {
    out[OUT_RESULT] = output.relinquish() ;
  }
}
