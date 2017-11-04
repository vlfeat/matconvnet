// @file vl_nnnormalizelp.cu
// @brief LP normalization MEX wrapper
// @author Sebastien Ehrhardt
// @author Andrea Vedaldi

/*
Copyright (C) 2017 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "bits/mexutils.h"
#include "bits/nnnormalizelp.hpp"
#include "bits/datamex.hpp"

#if ENABLE_GPU
#include "bits/datacu.hpp"
#endif

#include <cassert>
#include <vector>

using Int = vl::Int ;

/* ---------------------------------------------------------------- */
/*                                                       MEX driver */
/* ---------------------------------------------------------------- */

/* option codes */
enum {
  opt_verbose = 0,
  opt_epsilon,
  opt_dimensions,
  opt_exponent,
  opt_spatial,
  opt_norms,
  opt_cudnn,
  opt_no_cudnn,
} ;

/* options */
VLMXOption  options [] = {
  {"Verbose",          0,   opt_verbose           },
  {"Epsilon",	         1,   opt_epsilon           },
  {"Dimensions",       1,   opt_dimensions        },
  {"Exponent",         1,   opt_exponent          },
  {"P",                1,   opt_exponent          },
  {"Spatial",          0,   opt_spatial           },
  {"Norms",            1,   opt_norms             },
  {"Cudnn",            0,   opt_cudnn             },
  {"NoCudnn",          0,   opt_no_cudnn          },
  {0,                  0,   0                     }
} ;

enum {
  IN_DATA = 0, IN_DEROUTPUT, IN_END
} ;

enum {
  OUT_RESULT = 0,
  OUT_END
} ;

vl::ErrorCode performNormalizeLp(vl::MexContext& context,
                                 int nout, mxArray *out[],
                                 int nin, mxArray const *in[])
{
  bool backMode = false ;
  bool givenNormsMode = false ;
  bool returnNormsMode = false ;
  mxArray const* normsArray ;

  int verbosity = 0 ;
  int opt ;
  int next = IN_END ;
  mxArray const *optarg ;

  vl::nn::NormalizeLp op(context) ;

  /* -------------------------------------------------------------- */
  /*                                            Check the arguments */
  /* -------------------------------------------------------------- */

  if (nin < 1) {
    mexErrMsgTxt("There is less than one argument") ;
  }
  if (nin > 1 && vlmxIsString(in[1],-1)) {
    next = 1 ;
    backMode = 0 ;
  } else {
    backMode = (nin >= 2) ;
  }
  returnNormsMode = nout > 1 ;

  while ((opt = vlmxNextOption (in, nin, options, &next, &optarg)) >= 0) {
    switch (opt) {
      case opt_verbose : context.setLogLevel(++verbosity) ; break ;
      case opt_norms : normsArray = optarg ; givenNormsMode = true ; break ;
      case opt_epsilon : MXOPTDSCAL(EPSILON,setEpsilon) ; break ;
      case opt_exponent : MXOPTDSCAL(EXPONENT,setExponent) ; break ;
      case opt_dimensions : {
        std::vector<Int> dims ;
        if (context.parse(dims,optarg) != vl::VLE_Success) {
          return context.passError(vl::VLE_IllegalArgument, "Could not set SELECTEDDIMENSIONS:") ;
        }
        for (auto& dim : dims) dim-- ;
        MXCHECK(op.setSelectedDimensions(dims)) ;
        break ;
      }
      case opt_spatial : op.setSelectedDimensions({0,1}) ; break ;
      case opt_no_cudnn:
#if ENABLE_CUDNN
        context.getCudaHelper().setCudnnEnabled(false) ;
#endif
        break ;

      case opt_cudnn :
#if ENABLE_CUDNN
        context.getCudaHelper().setCudnnEnabled(true) ;
#endif
        break ;
        
      default:
        break ;
    }
  }

  vl::MexTensor data(context) ;
  data.init(in[IN_DATA]) ;
  data.reshape(4) ;

  vl::DeviceType deviceType = data.getDeviceType() ;
  vl::DataType dataType = data.getDataType() ;
  vl::TensorShape outputShape, normShape ;
  MXCHECK(op.forwardShape(outputShape,normShape,data)) ;

  vl::MexTensor norms(context) ;
  if (givenNormsMode) {
    norms.init(normsArray) ;
    norms.reshape(2) ;
  }
  else if (returnNormsMode) {
    norms.init(deviceType, dataType, normShape) ;
  }

  if (!backMode) {
    // Forward mode.
    // Initialize output tensor.
    vl::MexTensor output(context) ;
    output.init(deviceType, dataType, outputShape) ;

    // Perform calculation.
    if (!givenNormsMode) { MXCHECK(op.forward(output,norms,data)) ; }
    else { MXCHECK(op.forwardWithNorms(output,norms,data)) ; }

    // Return results.
    out[OUT_RESULT] = output.relinquish() ;
  }
  else {
    // Backward mode.
    vl::MexTensor derOutput(context) ;
    derOutput.init(in[IN_DEROUTPUT]) ;
    derOutput.reshape(4) ;

    // Initialize the tensors to be returned.
    vl::MexTensor derData(context) ;
    derData.init(deviceType, dataType, data.getShape()) ;

    // Perform calculation.
    if (!givenNormsMode) { MXCHECK(op.backward(derData,norms,data,derOutput)) ; }
    else { MXCHECK(op.backwardWithNorms(derData,norms,data,derOutput)) ; }

    // Return results.
    out[OUT_RESULT] = derData.relinquish() ;
  }
  out[OUT_RESULT+1] = norms.relinquish() ;
  return vl::VLE_Success ;
}

/* ---------------------------------------------------------------- */
/*                                                          Context */
/* ---------------------------------------------------------------- */

vl::MexContext context ;
void atExit() { context.clear() ; }

void mexFunction(int nout, mxArray *out[],
                 int nin, mxArray const *in[])
{
  mexAtExit(atExit) ;
  context.setLogLevel(0) ;
  context.clearLog() ;

  vl::ErrorCode error = performNormalizeLp(context,nout,out,nin,in) ;

  if (context.getLogLevel() > 0) {
    mexPrintf("vl_nnnormalizelp:\n") ;
    for (auto const & str : context.getLogbook()) {
      mexPrintf("\t%s\n", str.c_str()) ;
    }
    context.setLogLevel(0) ;
  }

  if (error != vl::VLE_Success) {
    vlmxError(VLMXE_IllegalArgument, context.getLastErrorMessage().c_str()) ;
  }
  return ;
}


