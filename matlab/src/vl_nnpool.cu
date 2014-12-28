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
#include "bits/nnhelper.h"
#include "bits/pooling.hpp"

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

VlEnumerator nnPoolMethodTypes [NN_POOL_METHODS_NUM] =
{
  {"Max",     (vl_index)NN_POOL_MAX     },
  {"Avg",     (vl_index)NN_POOL_AVG     }
} ;

enum {
  IN_DATA = 0, IN_SIZE, IN_DEROUTPUT, IN_END
} ;

enum {
  OUT_RESULT = 0, OUT_END
} ;

void mexFunction(int nout, mxArray *out[],
                 int nin, mxArray const *in[])
{
  /* inputs */
  PackedData data ;
  PackedData derOutput ;

  /* outputs */
  PackedData output ;
  PackedData derData  ;
  PackedDataGeometry outputGeom ;
  PackedDataGeometry derDataGeom  ;

  int poolWidth ;
  int poolHeight ;
  int strideX = 1 ;
  int strideY = 1 ;
  int padLeft = 0 ;
  int padRight = 0 ;
  int padTop = 0 ;
  int padBottom = 0 ;
  PoolMethod method = NN_POOL_MAX;

#ifdef ENABLE_GPU
  bool gpuMode = false ;
#else
  bool const gpuMode = false ;
#endif
  bool backMode = false ;

  int verbosity = 0 ;
  int opt ;
  int next = IN_END ;
  mxArray const *optarg ;
  VlEnumerator *pair ;

  packed_data_init_empty(&data) ;
  packed_data_init_empty(&derOutput) ;
  packed_data_init_empty(&output) ;
  packed_data_init_empty(&derData) ;

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
        pair = vlmxDecodeEnumeration(optarg, nnPoolMethodTypes, VL_TRUE) ;
        if (pair == NULL) {
          vlmxError(vlmxErrInvalidArgument, "METHOD is not a supported method.") ;
        }
        method = (PoolMethod)pair->value ;
        break;
      default: break ;
    }
  }

  packed_data_init_with_array(&data, in[IN_DATA]) ;
  if (backMode) { packed_data_init_with_array(&derOutput, in[IN_DEROUTPUT]) ; }

#if ENABLE_GPU
  gpuMode = (data.mode == matlabGpuArrayWrapper) ;
  if (gpuMode) {
    mxInitGPU() ;
  }
#endif

  /* check GPU/data class consistency */
  if (gpuMode && (derOutput.mode != matlabGpuArrayWrapper && backMode)) {
    mexErrMsgTxt("DATA is a GPU array but DEROUTPUT is not.") ;
  }
  if (data.geom.classID != mxSINGLE_CLASS) {
    mexErrMsgTxt("DATA is not of class SINGLE.");
  }
  if (backMode && (derOutput.geom.classID != mxSINGLE_CLASS)) {
    mexErrMsgTxt("DEROUTPUT is not of class SINGLE.");
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

  if (strideX < 1 || strideY < 1) {
    mexErrMsgTxt("At least one element of STRIDE is smaller than one.") ;
  }

  packed_data_geom_init(&outputGeom,
                        mxSINGLE_CLASS,
                        (data.geom.height + (padTop+padBottom) - poolHeight)/strideY + 1,
                        (data.geom.width + (padLeft+padRight) - poolWidth)/strideX + 1,
                        data.geom.depth,
                        data.geom.size) ;

  derDataGeom = data.geom ;

  if (verbosity > 0) {
    mexPrintf("vl_nnpool: mode %s; %s\n", gpuMode?"gpu":"cpu", backMode?"backward":"forward") ;
    mexPrintf("vl_nnpool: stride: [%d %d], pad: [%d %d %d %d]\n",
              strideY, strideX,
              padTop, padBottom, padLeft, padRight) ;
    packed_data_geom_display(&data.geom, "vl_nnpool: data") ;
    mexPrintf("vl_nnpool: pooling: %d x %d\n", poolHeight, poolWidth);
    mexPrintf("vl_nnpool: method: %s\n",
              vl_enumeration_get_by_value(nnPoolMethodTypes, method)->name);
    if (backMode) {
      packed_data_geom_display(&derOutput.geom, "vl_nnpool: derOutput") ;
      packed_data_geom_display(&derDataGeom, "vl_nnpool: derData") ;
    } else {
      packed_data_geom_display(&outputGeom, "vl_nnpool: output") ;
    }
  }

  if (backMode) {
    if (derOutput.geom.height != outputGeom.height ||
        derOutput.geom.width != outputGeom.width ||
        derOutput.geom.depth != outputGeom.depth ||
        derOutput.geom.size != outputGeom.size)
    {
      mexErrMsgTxt("DEROUTPUT dimensions are incompatible with X and POOL.") ;
    }
  }

  if (data.geom.height < poolHeight || data.geom.width < poolWidth) {
    mexErrMsgTxt("Pooling SIZE is larger than the DATA.") ;
  }

  if (poolHeight == 0 || poolWidth == 0) {
    mexErrMsgTxt("A dimension of the pooling SIZE is void.") ;
  }

  if (strideX == 0 || strideY == 0) {
    mexErrMsgTxt("An element of STRIDE is zero.") ;
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

  /* -------------------------------------------------------------- */
  /*                                                    Do the work */
  /* -------------------------------------------------------------- */

  if (!backMode) {
    packed_data_init_with_geom(&output, gpuMode, outputGeom, false, true, 0) ;
  } else {
    packed_data_init_with_geom(&derData, gpuMode, derDataGeom, false, true, 0) ;
  }

  if (backMode) {
    /* ---------------------------------------------------------- */
    /*                                              Backward mode */
    /* ---------------------------------------------------------- */
    if (gpuMode) {
#ifdef ENABLE_GPU
      poolingBackward_gpu<float>(derData.memory,
                                 data.memory,
                                 derOutput.memory,
                                 method,
                                 data.geom.height, data.geom.width,
                                 data.geom.depth * data.geom.size,
                                 poolHeight,
                                 poolWidth,
                                 strideY,
                                 strideX,
                                 padTop,
                                 padBottom,
                                 padLeft,
                                 padRight) ;
#else
      assert(false) ;
#endif
    } else {
      poolingBackward_cpu<float>(derData.memory,
                                 data.memory,
                                 derOutput.memory,
                                 method,
                                 data.geom.height, data.geom.width,
                                 data.geom.depth * data.geom.size,
                                 poolHeight,
                                 poolWidth,
                                 strideY,
                                 strideX,
                                 padTop,
                                 padBottom,
                                 padLeft,
                                 padRight) ;
    }
  } else {
    /* ---------------------------------------------------------- */
    /*                                               Forward mode */
    /* ---------------------------------------------------------- */
    if (gpuMode) {
#ifdef ENABLE_GPU
      pooling_gpu<float>(output.memory,
                         data.memory,
                         method,
                         data.geom.height, data.geom.width,
                         data.geom.depth * data.geom.size,
                         poolHeight,
                         poolWidth,
                         strideY,
                         strideX,
                         padTop,
                         padBottom,
                         padLeft,
                         padRight) ;
#else
      assert(false) ;
#endif
    } else {
      pooling_cpu<float>(output.memory,
                         data.memory,
                         method,
                         data.geom.height, data.geom.width,
                         data.geom.depth * data.geom.size,
                         poolHeight,
                         poolWidth,
                         strideY,
                         strideX,
                         padTop,
                         padBottom,
                         padLeft,
                         padRight) ;
    }
  }

  /* -------------------------------------------------------------- */
  /*                                                        Cleanup */
  /* -------------------------------------------------------------- */

  packed_data_deinit(&data) ;
  if (backMode) {
    packed_data_deinit(&derOutput) ;
    out[OUT_RESULT] = packed_data_deinit_extracting_array(&derData) ;
  } else {
    out[OUT_RESULT] = packed_data_deinit_extracting_array(&output) ;
  }
}
