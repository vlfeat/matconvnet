// @file vl_nnbilinearsampler.cu
// @brief Bilinear Sampler MEX wrapper
// @author Ankush Gupta
// @author Andrea Vedaldi
/*
Copyright (C) 2016 Ankush Gupta and Andrea Vedaldi.
Copyright (C) 2017 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "bits/mexutils.h"
#include "bits/datamex.hpp"
#include "bits/nnbilinearsampler.hpp"

#if ENABLE_GPU
#include "bits/datacu.hpp"
#endif

#include <cassert>

using Int = vl::Int ;

/* option codes */
enum {
  opt_verbose = 0,
  opt_cudnn,
  opt_no_cudnn
};

/* options */
VLMXOption  options [] = {
  {"Verbose",          0,   opt_verbose           },
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

#define ERR(code,message) \
context.passError(code,message)

#define CHECK2(x) \
{ vl::ErrorCode err = (x) ; if (err != vl::VLE_Success) { return err ; } }

/* ---------------------------------------------------------------- */
/*                                                       MEX driver */
/* ---------------------------------------------------------------- */

enum {
  IN_DATA = 0, IN_GRID, IN_DEROUTPUT, IN_END
} ;

enum {
  OUT_RESULT = 0, OUT_DERGRID, OUT_END
} ;

vl::ErrorCode
performBilinearSampler(vl::MexContext& context,
                       int nout, mxArray *out[],
                       int nin, mxArray const *in[])
{
  // whether we are back-propagating or not:
  bool backMode = false ;

  int verbosity = 0 ;
  int opt ;
  int next = IN_END ;
  mxArray const *optarg ;

  vl::nn::BilinearSampler op(context) ;

  // need at least data and grid to operate (2 args minimum)
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
        context.setLogLevel(verbosity) ;
        break ;
        
      case opt_no_cudnn :
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
  data.reshape(4) ; // -> 4 dimensions

  vl::MexTensor grid(context) ;
  grid.init(in[IN_GRID]);
  grid.reshape(4); // ->  4 dimensions

  if (!backMode) {
    // Forward mode.
    vl::DeviceType deviceType = data.getDeviceType() ;
    vl::DataType dataType = data.getDataType() ;

    // Compute the size of the output tensor.
    vl::TensorShape outputShape ;
    CHECK2(op.forwardShape(outputShape,data,grid)) ;

    // Get output tensors.
    vl::MexTensor output(context) ;
    output.initWithZeros(deviceType, dataType, outputShape) ;

    // Perform calculation.
    CHECK2(op.forward(output,data,grid)) ;
    out[OUT_RESULT] = output.relinquish() ;
  }
  else {
    // Backward mode.
    vl::MexTensor derOutput(context) ;
    derOutput.init(in[IN_DEROUTPUT]) ;
    derOutput.reshape(4) ; // -> 4 dimensions
    vl::DeviceType deviceType = derOutput.getDeviceType() ;
    vl::DataType dataType = derOutput.getDataType() ;

    // Get output tensors.
    vl::MexTensor derData(context) ;
    vl::MexTensor derGrid(context) ;
    derData.initWithZeros(deviceType, dataType, data.getShape()) ;
    derGrid.initWithZeros(deviceType, dataType, grid.getShape()) ;

    // Perform calculation.
    CHECK2(op.backward(derData,derGrid,data,grid,derOutput)) ;
    out[OUT_RESULT] = derData.relinquish() ;
    out[OUT_DERGRID] = derGrid.relinquish() ;
  }
  return vl::VLE_Success ;
}

void mexFunction(int nout, mxArray *out[],
                 int nin, mxArray const *in[])
{
  mexAtExit(atExit) ;
  context.setLogLevel(0) ;
  context.clearLog() ;

  vl::ErrorCode error = performBilinearSampler(context,nout,out,nin,in) ;

  if (context.getLogLevel() > 0) {
    mexPrintf("vl_nnbilinearsampler:\n") ;
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

