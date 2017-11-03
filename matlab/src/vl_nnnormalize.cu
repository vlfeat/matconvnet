// @file vl_nnnormalize.cu
// @brief Normalization block MEX wrapper
// @author Andrea Vedaldi

/*
Copyright (C) 2014-15 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "bits/mexutils.h"
#include "bits/nnnormalize.hpp"
#include "bits/datamex.hpp"

#if ENABLE_GPU
#include "bits/datacu.hpp"
#endif

#include <cassert>

using Int = vl::Int ;

/* option codes */
enum {
  opt_verbose = 0
} ;

/* options */
VLMXOption  options [] = {
  {"Verbose",          0,   opt_verbose           },
  {0,                  0,   0                     }
} ;

/* ---------------------------------------------------------------- */
/*                                                          Context */
/* ---------------------------------------------------------------- */

vl::MexContext context ;

/*
 Resetting the context here resolves a crash when MATLAB quits and
 the ~Context function is implicitly called on unloading the MEX file.
 */
void atExit()
{
  context.clear() ;
}

/* ---------------------------------------------------------------- */
/*                                                       MEX driver */
/* ---------------------------------------------------------------- */

enum {
  IN_DATA = 0, IN_PARAM, IN_DEROUTPUT, IN_END
} ;

enum {
  OUT_RESULT = 0, OUT_END
} ;

vl::ErrorCode performLRN(vl::MexContext& context,
                         int nout, mxArray *out[],
                         int nin, mxArray const *in[])
{
  bool backMode = false ;

  int verbosity = 0 ;
  int opt ;
  int next = IN_END ;
  mxArray const *optarg ;

  /* -------------------------------------------------------------- */
  /*                                            Check the arguments */
  /* -------------------------------------------------------------- */

  if (nin < 2) {
    return context.passError(vl::VLE_IllegalArgument, "There are less than two arguments.") ;
  }

  if (nin > 2 && vlmxIsString(in[2],-1)) {
    next = 2 ;
    backMode = 0 ;
  } else {
    backMode = (nin >= 3) ;
  }

  vl::nn::LRN op(context) ;

  if (!mxIsNumeric(in[IN_PARAM]) ||
      mxGetClassID(in[IN_PARAM]) != mxDOUBLE_CLASS ||
      mxIsComplex(in[IN_PARAM]) ||
      mxGetNumberOfElements(in[IN_PARAM]) != 4)
  {
    return context.setError(vl::VLE_IllegalArgument, "PARAM is not a plain 4 vector.")  ;
  }
  {
    double const * params = mxGetPr(in[IN_PARAM]) ;
    MXCHECK(op.setParameters((Int)params[0],params[1],params[2],params[3])) ;
  }

  while ((opt = vlmxNextOption (in, nin, options, &next, &optarg)) >= 0) {
    switch (opt) {
      case opt_verbose : context.setLogLevel(++verbosity) ; break ;
      default: break ;
    }
  }

  vl::MexTensor data(context) ;
  data.init(in[IN_DATA]) ;
  data.reshape(4) ;

  if (!backMode) {
    // Forward mode.
    vl::DeviceType deviceType = data.getDeviceType() ;
    vl::DataType dataType = data.getDataType() ;

    // Compute the size of the output tensor.
    vl::TensorShape outputShape ;
    MXCHECK(op.forwardShape(outputShape,data)) ;

    // Initialize output tensor.
    vl::MexTensor output(context) ;
    output.init(deviceType, dataType, outputShape) ;

    // Perform calculation.
    MXCHECK(op.forward(output,data)) ;

    // Return results.
    out[OUT_RESULT] = output.relinquish() ;
  } else {
    // Backward mode.
    vl::MexTensor derOutput(context) ;
    derOutput.init(in[IN_DEROUTPUT]) ;
    derOutput.reshape(4) ;
    vl::DeviceType deviceType = derOutput.getDeviceType() ;
    vl::DataType dataType = derOutput.getDataType() ;

    // Initialize the tensors to be returned.
    vl::MexTensor derData(context) ;
    derData.init(deviceType, dataType, data.getShape()) ;

    // Perform calculation.
    MXCHECK(op.backward(derData,data,derOutput)) ;

    // Return results.
    out[OUT_RESULT] = derData.relinquish() ;
  }
  return vl::VLE_Success ;
}

void mexFunction(int nout, mxArray *out[],
                 int nin, mxArray const *in[])
{
  mexAtExit(atExit) ;
  context.setLogLevel(0) ;
  context.clearLog() ;

  vl::ErrorCode error = performLRN(context,nout,out,nin,in) ;

  if (context.getLogLevel() > 0) {
    mexPrintf("vl_nnnormalize:\n") ;
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


