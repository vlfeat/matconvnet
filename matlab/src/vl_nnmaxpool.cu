/** @file vl_nnmaxpool.cu
 ** @brief Max-pooling block
 ** @author Andrea Vedaldi
 **/

/*
Copyright (C) 2014 Andrea Vedaldi.
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

#if ENABLE_GPU
  gpuMode = mxIsGPUArray(in[IN_DATA]) ;
  if (gpuMode) {
    mxInitGPU() ;
  }
#else
  if (!mxIsNumeric(in[IN_DATA])) {
    mexErrMsgTxt("DATA must be numeric (note: GPU support not compiled).") ;
  }
#endif

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
            mexErrMsgTxt("STRIDE has neither one nor two elements.") ;
        }

      default: break ;
    }
  }

  packed_data_init_with_array (&data, gpuMode, in[IN_DATA]) ;
  if (backMode) {
    packed_data_init_with_array(&derOutput, gpuMode, in[IN_DEROUTPUT]) ;
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

  packed_data_geom_init(&outputGeom,
                        mxSINGLE_CLASS,
                        (data.geom.height + (padTop+padBottom) - poolHeight)/strideY + 1,
                        (data.geom.width + (padLeft+padRight) - poolWidth)/strideX + 1,
                        data.geom.depth,
                        data.geom.size) ;

  derDataGeom = data.geom ;

  if (verbosity > 0) {
    mexPrintf("vl_nnmaxpool: mode %s; %s\n", gpuMode?"gpu":"cpu", backMode?"backward":"forward") ;
    mexPrintf("vl_nnmaxpool: stride: [%d %d], pad: [%d %d %d %d]\n",
              strideY, strideX,
              padTop, padBottom, padLeft, padRight) ;
    packed_data_geom_display(&data.geom, "vl_nnmaxpool: data") ;
    mexPrintf("vl_nnmaxpool: pooling: %d x %d\n", poolHeight, poolWidth);
    if (backMode) {
      packed_data_geom_display(&derOutput.geom, "vl_nnmaxpool: derOutput") ;
      packed_data_geom_display(&derDataGeom, "vl_nnmaxpool: derData") ;
    } else {
      packed_data_geom_display(&outputGeom, "vl_nnmaxpool: output") ;
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

  for (int image = 0 ; image < data.geom.size ; ++image) {
    ptrdiff_t dataOffset = (data.geom.height*data.geom.width*data.geom.depth) * image ;
    ptrdiff_t outputOffset = (output.geom.height*output.geom.width*output.geom.depth) * image ;
    ptrdiff_t derOutputOffset = (derOutput.geom.height*derOutput.geom.width*derOutput.geom.depth) * image ;

    if (backMode) {
      /* ---------------------------------------------------------- */
      /*                                              Backward mode */
      /* ---------------------------------------------------------- */
      if (gpuMode) {
#ifdef ENABLE_GPU
        maxPoolingBackward_gpu<float>(derData.memory + dataOffset,
                                      data.memory + dataOffset,
                                      derOutput.memory + derOutputOffset,
                                      data.geom.height, data.geom.width, data.geom.depth,
                                      poolWidth, strideX, padLeft) ;
#else
        assert(false) ;
#endif
      } else {
        maxPoolingBackward_cpu<float>(derData.memory + dataOffset,
                                      data.memory + dataOffset,
                                      derOutput.memory + derOutputOffset,
                                      data.geom.height, data.geom.width, data.geom.depth,                             
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
        maxPooling_gpu<float>(output.memory + outputOffset,
                              data.memory + dataOffset,
                              data.geom.height, data.geom.width, data.geom.depth,
                              poolWidth, strideX, padLeft) ;
#else
        assert(false) ;
#endif
      } else {
        maxPooling_cpu<float>(output.memory + outputOffset,
                              data.memory + dataOffset,
                              data.geom.height, data.geom.width, data.geom.depth,
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
