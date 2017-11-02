// @file vl_nnbnorm.cu
// @brief Batch normalization MEX wrapper
// @author Sebastien Ehrhardt
// @author Andrea Vedaldi

/*
Copyright (C) 2015 Sebastien Ehrhardt and Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "bits/mexutils.h"
#include "bits/nnbnorm.hpp"
#include "bits/datamex.hpp"

#if ENABLE_GPU
#include "bits/datacu.hpp"
#endif

#include <cassert>

using namespace vl ;

/* option codes */
enum {
  opt_verbose = 0,
  opt_epsilon,
  opt_moments,
  opt_cudnn,
  opt_no_cudnn,
} ;

/* options */
VLMXOption  options [] = {
  {"Verbose",          0,   opt_verbose           },
  {"Epsilon",	         1,   opt_epsilon           },
  {"Moments",          1,   opt_moments           },
  {"Cudnn",            0,   opt_cudnn             },
  {"NoCudnn",          0,   opt_no_cudnn          },
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
  IN_DATA = 0, IN_MULTIPLIERS, IN_BIASES, IN_DEROUTPUT, IN_END
} ;

enum {
  OUT_RESULT = 0,
  OUT_DERMULTIPLIERS,
  OUT_DERBIASES,
  OUT_END
} ;

vl::ErrorCode
performBatchNorm(vl::MexContext& context,
                 int nout, mxArray *out[],
                 int nin, mxArray const *in[])
{
  bool backMode = false ;
  bool givenMomentsMode = false ;
  mxArray const * momentsArray ;

  int verbosity = 0 ;
  int opt ;
  int next = IN_END ;
  mxArray const *optarg ;

  /* -------------------------------------------------------------- */
  /*                                            Check the arguments */
  /* -------------------------------------------------------------- */

  if (nin < 3) {
    return context.setError(VLE_IllegalArgument, "The arguments are less than three.") ;
  }
  if (nin > 3 && vlmxIsString(in[3],-1)) {
    next = 3 ;
    backMode = 0 ;
  } else {
    backMode = (nin >= 4) ;
  }
  bool returnMomentsMode = backMode ? (nout > 3) : (nout > 1) ;

  vl::nn::BatchNorm op(context) ;

  while ((opt = vlmxNextOption (in, nin, options, &next, &optarg)) >= 0) {
    switch (opt) {
      case opt_verbose : context.setLogLevel(++verbosity) ; break ;
      case opt_epsilon : MXOPTDSCAL(EPSILON,setEpsilon) ; break ;
      case opt_moments : momentsArray = optarg ; givenMomentsMode = true ; break ;
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

  vl::MexTensor multipliers(context);
  multipliers.init(in[IN_MULTIPLIERS]) ;
  multipliers.reshape(1) ;

  vl::MexTensor biases(context);
  biases.init(in[IN_BIASES]) ;
  biases.reshape(1) ;

  // Compute the size of the output tensors.
  vl::DeviceType deviceType = data.getDeviceType() ;
  vl::DataType dataType = data.getDataType() ;
  vl::TensorShape outputShape ;
  vl::TensorShape momentShape ;
  MXCHECK(op.forwardShape(outputShape,momentShape,data)) ;

  // Get moments.
  vl::MexTensor moments(context) ;
  if (givenMomentsMode) {
    moments.init(momentsArray) ;
    moments.reshape(2) ;
  } else {
    if (returnMomentsMode) {
      moments.init(deviceType, dataType, {data.getNumChannels(), 2, 1, 1}) ;
    }
  }

  if (!backMode) {
    // Forward mode.
    // Initialize output tensor.
    vl::MexTensor output(context) ;
    output.init(deviceType, dataType, outputShape) ;

    // Perform calculation.
    if (givenMomentsMode) {
      MXCHECK(op.forwardWithMoment(output,moments,data,multipliers,biases)) ;
    } else {
      MXCHECK(op.forward(output,moments,data,multipliers,biases)) ;
    }

    // Return results.
    out[OUT_RESULT] = output.relinquish() ;
    out[1] = moments.relinquish() ;
  } else {
    // Backward mode.
    vl::MexTensor derOutput(context) ;
    derOutput.init(in[IN_DEROUTPUT]) ;
    derOutput.reshape(4) ;

    // Initialize output tensors.
    vl::MexTensor derData(context) ;
    derData.init(deviceType, dataType, outputShape) ;

    vl::MexTensor derMultiplier(context) ;
    derMultiplier.init(deviceType, dataType, multipliers.getShape()) ;

    vl::MexTensor derBias(context) ;
    derBias.init(deviceType, dataType, biases.getShape()) ;

    // Perform calculation.
    if (givenMomentsMode) {
      MXCHECK(op.backwardWithMoment
             (derData,derMultiplier,derBias,moments,data,multipliers,biases,derOutput)) ;
    } else {
      MXCHECK(op.backward
             (derData,derMultiplier,derBias,moments,data,multipliers,biases,derOutput)) ;
    }
    out[OUT_RESULT] = derData.relinquish() ;
    out[OUT_DERMULTIPLIERS] = derMultiplier.relinquish() ;
    out[OUT_DERBIASES] = derBias.relinquish() ;
  }
  if (moments) {
    out[backMode ? 3 : 1] = moments.relinquish() ;
  }
  return VLE_Success ;
}

void mexFunction(int nout, mxArray *out[],
                 int nin, mxArray const *in[])
{
  mexAtExit(atExit) ;
  context.setLogLevel(0) ;
  context.clearLog() ;

  vl::ErrorCode error = performBatchNorm(context,nout,out,nin,in) ;

  if (context.getLogLevel() > 0) {
    mexPrintf("vl_nnbnorm:\n") ;
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

