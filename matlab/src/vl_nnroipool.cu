// @file vl_nnroipooling.cpp
// @brief ROI pooling block
// @author Hakan Bilen
// @author Abishek Dutta
// @author Andrea Vedaldi

/*
Copyright (C) 2016-17 Hakan Bilen, Abishek Dutta, and Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "bits/mexutils.h"
#include "bits/datamex.hpp"
#include "bits/nnroipooling.hpp"

#if ENABLE_GPU
#include "bits/datacu.hpp"
#endif

#include <cassert>
#include <algorithm>

using Int = vl::Int ;

/* option codes */
enum {
  opt_method = 0,
  opt_subdivisions,
  opt_transform,
  opt_verbose,
} ;

/* options */
VLMXOption  options [] = {
  {"Method",           1,   opt_method       },
  {"Subdivisions",     1,   opt_subdivisions },
  {"Transform",        1,   opt_transform    },
  {"Verbose",          0,   opt_verbose      },
  {0,                  0,   0                }
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
  IN_DATA = 0, IN_ROIS, IN_DEROUTPUT, IN_END
} ;

enum {
  OUT_RESULT = 0, OUT_END
} ;

vl::ErrorCode performROIPooling(vl::MexContext& context,
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

  vl::nn::ROIPooling op(context) ;

  while ((opt = vlmxNextOption (in, nin, options, &next, &optarg)) >= 0) {
    switch (opt) {
      case opt_verbose : context.setLogLevel(++verbosity) ; break ;
      case opt_subdivisions : MXOPTIVEC(SUBDIVISIONS,setSubdivisions) ; break ;
      case opt_transform: MXOPTDVEC(TRANSFORM,setTransform) ; break ;
      case opt_method : {
        if (!vlmxIsString(optarg,-1)) {
          vlmxError(VLMXE_IllegalArgument, "METHOD is not a string.") ;
        }
        if (vlmxIsEqualToStringI(optarg, "max")) { op.setMethod(vl::nn::ROIPooling::Max) ; }
        else if (vlmxIsEqualToStringI(optarg, "avg")) { op.setMethod(vl::nn::ROIPooling::Average) ; }
        else context.setError(vl::VLE_IllegalArgument, "METHOD is not a supported method.") ;
        break ;
      }
      default: assert(false) ; break ;
    }
  }

  vl::MexTensor data(context) ;
  data.init(in[IN_DATA]) ;

  vl::MexTensor rois(context) ;
  rois.init(in[IN_ROIS]);

  if (!backMode) {
    // Forward mode.
    vl::DeviceType deviceType = data.getDeviceType() ;
    vl::DataType dataType = data.getDataType() ;

    // Compute the size of the output tensor.
    vl::TensorShape outputShape ;
    MXCHECK(op.forwardShape(outputShape,data,rois)) ;

    // Initialize output tensor.
    vl::MexTensor output(context) ;
    output.initWithZeros(deviceType, dataType, outputShape) ;

    // Perform calculation.
    MXCHECK(op.forward(output,data,rois)) ;

    // Return results.
    out[OUT_RESULT] = output.relinquish() ;
  } else {
    // Backward mode.
    vl::MexTensor derOutput(context) ;
    derOutput.init(in[IN_DEROUTPUT]) ;
    vl::DeviceType deviceType = derOutput.getDeviceType() ;
    vl::DataType dataType = derOutput.getDataType() ;

    // Initialize the tensors to be returned.
    vl::MexTensor derData(context) ;
    derData.initWithZeros(deviceType, dataType, data.getShape()) ;

    // Perform calculation.
    MXCHECK(op.backward(derData,data,rois,derOutput)) ;

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

  vl::ErrorCode error = performROIPooling(context,nout,out,nin,in) ;

  if (context.getLogLevel() > 0) {
    mexPrintf("vl_nnroipoool:\n") ;
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

#if 0

  Int numROIs = rois.getNumElements() / 5 ;

  if (! vl::areCompatible(data, rois)) {
    vlmxError(VLMXE_IllegalArgument, "DATA and ROI do not have compatible formats.") ;
  }

  if (rois.getNumElements() != numROIs * 5 || numROIs == 0) {
    vlmxError(VLMXE_IllegalArgument, "ROI is not a 5 x K array with K >= 1.") ;
  }
  rois.reshape(vl::TensorShape(1, 1, 5, numROIs)) ;

  vl::TensorShape dataShape = data.getShape();
  dataShape.reshape(4);

  /* Get the output geometry */
  vl::TensorShape outputShape(subdivisions[0],
                              subdivisions[1],
                              dataShape.getNumChannels(),
                              numROIs) ;

  vl::TensorShape derOutputShape = derOutput.getShape();
  /* in case there is only one roi */ 
  derOutputShape.reshape(4);

  if (backMode) {
    if (derOutputShape != outputShape) {
      vlmxError(VLMXE_IllegalArgument, "The shape of DEROUTPUT is incorrect.") ;
    }
  }

  /* Create output buffers */
  vl::DeviceType deviceType = data.getDeviceType() ;
  vl::DataType dataType = data.getDataType() ;
  vl::MexTensor output(context) ;
  vl::MexTensor derData(context) ;

  if (!backMode) {
    output.initWithZeros(deviceType, dataType, outputShape) ;
  } else {
    derData.initWithZeros(deviceType, dataType, dataShape) ;
  }

  if (verbosity > 0) {
    mexPrintf("vl_nnroipool: %s; %s", backMode?"backward":"forward", (data.getDeviceType()==vl::VLDT_GPU) ? "GPU" : "CPU") ;
    mexPrintf("\nvl_nnroipool: method: %d; num ROIs: %d\n", method, numROIs);
    mexPrintf("vl_nnroipool: subdivisions: [%d x %d]\n", subdivisions[0], subdivisions[1]) ;
    mexPrintf("vl_nnroipool: transform: [%g %g %g ; %g %g %g]\n",
              transform[0], transform[2], transform[4],
              transform[1], transform[3], transform[5]) ;

    vl::print("vl_nnroipool: data: ", data) ;
    if (backMode) {
      vl::print("vl_nnroipool: derOutput: ", derOutput) ;
      vl::print("vl_nnroipool: derData: ", derData) ;
    } else {
      vl::print("vl_nnroipool: output: ", output) ;
      vl::print("vl_nnroipool: rois: ", rois) ;
    }
  }

  /* -------------------------------------------------------------- */
  /*                                                    Do the work */
  /* -------------------------------------------------------------- */

  vl::ErrorCode error ;
  vl::nn::ROIPooling op(context,subdivisions,transform,method) ;

  if (!backMode) {
    error = op.forward(output, data, rois) ;
  } else {
    error = op.backward(derData, data, rois, derOutput) ;
  }

  /* -------------------------------------------------------------- */
  /*                                                         Finish */
  /* -------------------------------------------------------------- */

  if (error != vl::VLE_Success) {
    vlmxError(VLMXE_IllegalArgument, context.getLastErrorMessage().c_str()) ;
  }
  if (backMode) {
    out[OUT_RESULT] = derData.relinquish() ;
  } else {
    out[OUT_RESULT] = output.relinquish() ;
  }
}
#endif
