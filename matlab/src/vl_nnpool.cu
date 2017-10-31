// @file vl_nnpool.cu
// @brief Pooling block MEX wrapper
// @author Andrea Vedaldi
// @author Karel Lenc

/*
Copyright (C) 2014-15 Andrea Vedaldi and Karel Lenc.
Copyright (C) 2017 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "bits/mexutils.h"
#include "bits/datamex.hpp"
#include "bits/nnpooling.hpp"

#if ENABLE_GPU
#include "bits/datacu.hpp"
#endif

#include <cassert>

using Int = vl::Int ;

/* option codes */
enum {
  opt_stride = 0,
  opt_padding,
  opt_shape,
  opt_method,
  opt_verbose,
  opt_cudnn,
  opt_no_cudnn,
} ;

/* options */
VLMXOption  options [] = {
  {"Stride",           1,   opt_stride            },
  {"Pad",              1,   opt_padding           },
  {"Padding",          1,   opt_padding           },
  {"Shape",            1,   opt_shape             },
  {"Method",           1,   opt_method            },
  {"Verbose",          0,   opt_verbose           },
  {"CUDNN",            0,   opt_cudnn             },
  {"NoCUDNN",          0,   opt_no_cudnn          },
  {0,                  0,   0                     }
} ;

/* ---------------------------------------------------------------- */
/*                                                          Context */
/* ---------------------------------------------------------------- */

vl::MexContext context ;

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
  IN_DATA = 0, IN_SIZE, IN_DEROUTPUT, IN_END
} ;

enum {
  OUT_RESULT = 0, OUT_END
} ;

vl::ErrorCode
performPooling(vl::Context& contetx,
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
    return ERR(vl::VLE_IllegalArgument, "There are less than two arguments.") ;
  }

  if (nin > 2 && vlmxIsString(in[2],-1)) {
    next = 2 ;
    backMode = 0 ;
  } else {
    backMode = (nin >= 3) ;
  }

  vl::nn::Pooling op(context) ;

  {
    // Set the pooling window shape.
    std::vector<Int> shape ;
    if (context.parse(shape,in[IN_SIZE]) != vl::VLE_Success) {
      return ERR(vl::VLE_IllegalArgument, "Could not set SHAPE:") ;
    }
    CHECK2(op.setShape(shape)) ;
  }

  while ((opt = vlmxNextOption (in, nin, options, &next, &optarg)) >= 0) {
    switch (opt) {
      case opt_verbose :
        ++ verbosity ;
        context.setLogLevel(verbosity) ;
        break ;

      case opt_stride : {
        std::vector<Int> stride ;
        if (context.parse(stride,optarg) != vl::VLE_Success) {
          return ERR(vl::VLE_IllegalArgument, "Could not set STRIDE:") ;
        }
        CHECK2(op.setStride(stride)) ;
        break ;
      }

      case opt_padding : {
        std::vector<Int> padding ;
        if (context.parse(padding,optarg) != vl::VLE_Success) {
          return ERR(vl::VLE_IllegalArgument, "Could not set PADDING:") ;
        }
        CHECK2(op.setPadding(padding)) ;
        break ;
      }

      case opt_shape: {
        // Alternative way of setting the pooling window shape.
        std::vector<Int> shape ;
        if (context.parse(shape,optarg) != vl::VLE_Success) {
          return ERR(vl::VLE_IllegalArgument, "Could not set SHAPE:") ;
        }
        CHECK2(op.setShape(shape)) ;
        break ;
      }

      case opt_method :
        if (!vlmxIsString(optarg,-1)) {
           return context.setError(vl::VLE_IllegalArgument, "METHOD is not a string.") ;
        }
        if (vlmxIsEqualToStringI(optarg, "max")) {
          op.setMethod(vl::nn::Pooling::Max) ;
        } else if (vlmxIsEqualToStringI(optarg, "avg")) {
          op.setMethod(vl::nn::Pooling::Average) ;
        } else {
          return context.setError(vl::VLE_IllegalArgument,
                                  "The value of METHOD is not a supported method.") ;
        }
        break;

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
  data.reshape(4) ;

  if (!backMode) {
    // Forward mode.
    vl::DeviceType deviceType = data.getDeviceType() ;
    vl::DataType dataType = data.getDataType() ;

    // Compute the size of the output tensor.
    vl::TensorShape outputShape ;
    CHECK2(op.forwardShape(outputShape,data)) ;

    // Initialize output tensor.
    vl::MexTensor output(context) ;
    output.initWithZeros(deviceType, dataType, outputShape) ;

    // Perform calculation.
    CHECK2(op.forward(output,data)) ;

    // Return results.
    out[OUT_RESULT] = output.relinquish() ;
  }
  else {
    // Backward mode.
    vl::MexTensor derOutput(context) ;
    derOutput.init(in[IN_DEROUTPUT]) ;
    derOutput.reshape(4) ;
    vl::DeviceType deviceType = derOutput.getDeviceType() ;
    vl::DataType dataType = derOutput.getDataType() ;

    // Initialize the tensors to be returned.
    vl::MexTensor derData(context) ;
    derData.initWithZeros(deviceType, dataType, data.getShape()) ;
    derData.reshape(4) ;

    // Perform calculation.
    CHECK2(op.backward(derData,data,derOutput)) ;

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

  vl::ErrorCode error = performPooling(context,nout,out,nin,in) ;

  if (context.getLogLevel() > 0) {
    mexPrintf("vl_nnpool:\n") ;
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

