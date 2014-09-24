/** @file gnormalize.cu
 ** @brief Normalization block
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
  IN_DATA = 0, IN_PARAM, IN_DEROUTPUT, IN_END
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

  size_t normDepth ;
  double normAlpha ;
  double normKappa ;
  double normBeta ;

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

  packed_data_init_with_array (&data, in[IN_DATA]) ;
  if (backMode) { packed_data_init_with_array(&derOutput, in[IN_DEROUTPUT]) ; }

#if ENABLE_GPU
  gpuMode = (data.mode == matlabGpuArrayWrapper) ;
  if (gpuMode) {
    mxInitGPU() ;
  }
#endif

  /* check GPU/data class consistency */
  if (gpuMode && (derOutput.mode != matlabGpuArrayWrapper & backMode)) {
    mexErrMsgTxt("DATA is a GPU array but DEROUTPUT is not.") ;
  }
  if (data.geom.classID != mxSINGLE_CLASS) {
    mexErrMsgTxt("DATA is not of class SINGLE.");
  }
  if (backMode && (derOutput.geom.classID != mxSINGLE_CLASS)) {
    mexErrMsgTxt("DEROUTPUT is not of class SINGLE.");
  }

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

  packed_data_geom_init(&outputGeom,
                        mxSINGLE_CLASS,
                        data.geom.height,
                        data.geom.width,
                        data.geom.depth,
                        data.geom.size) ;

  derDataGeom = data.geom ;

  if (verbosity > 0) {
    mexPrintf("vl_nnnormalize: mode %s; %s\n", gpuMode?"gpu":"cpu", backMode?"backward":"forward") ;
    mexPrintf("vl_nnnormalize: (depth,kappa,alpha,beta): (%d,%g,%g,%g)\n",
              normDepth, normKappa, normAlpha, normBeta) ;
    packed_data_geom_display(&data.geom, "vl_nnnormalize: data") ;

    if (backMode) {
      packed_data_geom_display(&derOutput.geom, "vl_nnnormalize: derOutput") ;
      packed_data_geom_display(&derDataGeom, "vl_nnnormalize: derData") ;
    } else {
      packed_data_geom_display(&outputGeom, "vl_nnnormalize: output") ;
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

  if (normDepth < 1) {
    mexErrMsgTxt("The normalization depth is smaller than 1.") ;
  }

  /* -------------------------------------------------------------- */
  /*                                                    Do the work */
  /* -------------------------------------------------------------- */

  if (!backMode) {
    packed_data_init_with_geom(&output, gpuMode, outputGeom, false, true, 0) ;
    //packed_data_init_with_geom(&output, gpuMode, outputGeom, false, false, 0) ;
  } else {
    packed_data_init_with_geom(&derData, gpuMode, derDataGeom, false, true, 0) ;
  }

  if (!backMode) {
    /* forward */
    if (gpuMode) {
#ifdef ENABLE_GPU
      normalize_gpu<float>(output.memory,
                           data.memory,
                           data.geom.height, data.geom.width, data.geom.depth, data.geom.size,
                           normDepth, normKappa, normAlpha, normBeta) ;
#else
      assert(false) ;
#endif
    } else {
      normalize_cpu<float>(output.memory,
                           data.memory,
                           data.geom.height, data.geom.width, data.geom.depth, data.geom.size,
                           normDepth, normKappa, normAlpha, normBeta) ;
    }
  } else {
    /* backward */
    if (gpuMode) {
#ifdef ENABLE_GPU
      normalizeBackward_gpu<float>(derData.memory,
                                   data.memory,
                                   derOutput.memory,
                                   data.geom.height, data.geom.width, data.geom.depth, data.geom.size,
                                   normDepth, normKappa, normAlpha, normBeta) ;
#else
      assert(false) ;
#endif
    } else {
      normalizeBackward_cpu<float>(derData.memory,
                                   data.memory,
                                   derOutput.memory,
                                   data.geom.height, data.geom.width, data.geom.depth, data.geom.size,
                                   normDepth, normKappa, normAlpha, normBeta) ;
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
