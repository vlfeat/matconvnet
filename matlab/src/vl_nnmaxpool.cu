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

  int stride = 1 ;
  int pad = 0 ;
  int poolWidth ;
  int poolHeight ;

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

  if (!mxIsNumeric(in[IN_SIZE]) ||
      mxGetClassID(in[IN_SIZE]) != mxDOUBLE_CLASS ||
      mxIsComplex(in[IN_SIZE]) ||
      mxGetNumberOfElements(in[IN_SIZE]) != 2)
  {
    mexErrMsgTxt("SIZE is not a plain 2 vector.") ;
  }
  poolHeight = mxGetPr(in[IN_SIZE])[0] ;
  poolWidth = mxGetPr(in[IN_SIZE])[1] ;

  packed_data_geom_init(&outputGeom,
                        mxSINGLE_CLASS,
                        (data.geom.height + 2*pad - poolHeight)/stride + 1,
                        (data.geom.width + 2*pad - poolWidth)/stride + 1,
                        data.geom.depth,
                        data.geom.size) ;

  derDataGeom = data.geom ;

  if (verbosity > 0) {
    mexPrintf("vl_nnmaxpool: mode %s; %s\n", gpuMode?"gpu":"cpu", backMode?"backward":"forward") ;
    mexPrintf("vl_nnmaxpool: stride: %d, pad: %d\n", stride, pad) ;
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

  if (poolHeight != poolWidth) {
    mexErrMsgTxt("Non-square pooling not supported yet.");
  }

  if (data.geom.height < poolHeight || data.geom.width < poolWidth) {
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
                                      poolWidth, stride, pad) ;
#else
        assert(false) ;
#endif
      } else {
        maxPoolingBackward_cpu<float>(derData.memory + dataOffset,
                                      data.memory + dataOffset,
                                      derOutput.memory + derOutputOffset,
                                      data.geom.height, data.geom.width, data.geom.depth,
                                      poolWidth, stride, pad) ;
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
                              poolWidth, stride, pad) ;
#else
        assert(false) ;
#endif
      } else {
        maxPooling_cpu<float>(output.memory + outputOffset,
                              data.memory + dataOffset,
                              data.geom.height, data.geom.width, data.geom.depth,
                              poolWidth, stride, pad) ;
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
