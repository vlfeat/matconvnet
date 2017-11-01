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

#define ERR(code,message) \
context.passError(code,message)

#define CHECK2(x) \
{ vl::ErrorCode err = (x) ; if (err != vl::VLE_Success) { return err ; } }

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
    mexErrMsgTxt("The arguments are less than three.") ;
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

      case opt_verbose :
        ++ verbosity ;
        break ;

      case opt_epsilon :
        if (!vlmxIsPlainScalar(optarg)) {
          return ERR(vl::VLE_IllegalArgument, "EPSILON is not a plain scalar.") ;
        }
        CHECK2(op.setEpsilon(mxGetPr(optarg)[0])) ;
        break ;

      case opt_moments:
        momentsArray = optarg ;
        givenMomentsMode = true ;
        break ;

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
  CHECK2(op.forwardShape(outputShape,momentShape,data)) ;

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
      CHECK2(op.forwardWithMoment(output,moments,data,multipliers,biases)) ;
    } else {
      CHECK2(op.forward(output,moments,data,multipliers,biases)) ;
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
      CHECK2(op.backwardWithMoment
             (derData,derMultiplier,derBias,moments,data,multipliers,biases,derOutput)) ;
    } else {
      CHECK2(op.backward
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


#if 0


  if (givenMomentsMode) {
    moments.init(momentsArray) ;
    moments.reshape(2) ;
  }

  /* Check for GPU/data class consistency */
  if (! vl::areCompatible(data, multipliers)) {
    mexErrMsgTxt("DATA and MULTIPLIERS do not have compatible formats.") ;
  }
  if (! vl::areCompatible(data, biases)) {
    mexErrMsgTxt("DATA and BIASES do not have compatible formats.") ;
  }
  if (backMode && ! vl::areCompatible(data, derOutput)) {
    mexErrMsgTxt("DATA and DEROUTPUT do not have compatible formats.") ;
  }
  if (backMode && (data.getShape() != derOutput.getShape())) {
    mexErrMsgTxt("DATA and DEROUTPUT do not have the same size.") ;
  }
  if (givenMomentsMode && ! vl::areCompatible(data, moments))
  {
    mexErrMsgTxt("DATA and MOMENTS do not have compatible formats.") ;
  }

  /* Get the filter geometry */
  vl::TensorShape multipliersGeom(multipliers) ;
  if (multipliersGeom.getHeight() != data.getNumChannels()) {
    mexErrMsgTxt("The MULTIPLIERS size does not match the DATA depth.") ;
  }
  vl::TensorShape biasesGeom(biases);
  if (biasesGeom.getHeight() != data.getNumChannels()) {
    mexErrMsgTxt("The BIASES size does not match the DATA depth.") ;
  }
  if (givenMomentsMode) {
    vl::TensorShape momentsGeom(moments) ;
    if (momentsGeom.getNumElements() != 2*data.getNumChannels()) {
      mexErrMsgTxt("The MOMENTS size does not match the DATA depth.") ;
    }
  }

  /* Create output buffers */
  vl::DeviceType deviceType = data.getDeviceType() ;
  vl::DataType dataType = data.getDataType() ;
  vl::MexTensor output(context) ;
  vl::MexTensor derData(context) ;
  vl::MexTensor derMultipliers(context) ;
  vl::MexTensor derBiases(context) ;

  if (returnMomentsMode & !givenMomentsMode) {
    vl::TensorShape momentsGeom(data.getNumChannels(), 2, 1, 1) ;
    moments.init(deviceType, dataType, momentsGeom) ;
  }

  if (!backMode) {
    output.init(deviceType, dataType, data.getShape()) ;
  } else {
    if (computeDerData) {
      derData.init(deviceType, dataType, data.getShape()) ;
    }
    if (computeDerMultipliers) {
      derMultipliers.init(deviceType, dataType, multipliers.getShape()) ;
    }
    if (computeDerBiases) {
      derBiases.init(deviceType, dataType, biases.getShape()) ;
    }
  }

  if (verbosity > 0) {
    mexPrintf("vl_nnbnorm: mode %s; %s; moments %s/%s\n",
              (data.getDeviceType()==vl::VLDT_GPU)?"gpu":"cpu",
              backMode?"backward":"forward",
              givenMomentsMode?"given":"computed",
              returnMomentsMode?"returned":"discared") ;
    vl::print("vl_nnbnorm: data: ", data) ;
    vl::print("vl_nnbnorm: multipliers: ", multipliers) ;
    vl::print("vl_nnbnorm: biases: ", biases) ;
    if (backMode) {
      vl::print("vl_nnbnorm: derOutput: ", derOutput) ;
      vl::print("vl_nnbnorm: derData: ", derData) ;
      vl::print("vl_nnbnorm: derMultipliers: ", derMultipliers) ;
      vl::print("vl_nnbnorm: derBiases: ", derBiases) ;
    } else {
      vl::print("vl_nnbnorm: output: ", output) ;
    }
    if (moments) { vl::print("vl_nnbnorm: moments: ", moments) ; }
    mexPrintf("vl_nnbnorm: epsilon: %f\n", epsilon) ;
  }

  /* -------------------------------------------------------------- */
  /*                                                    Do the work */
  /* -------------------------------------------------------------- */

  vl::ErrorCode error ;
  vl::nn::BatchNorm op(context,epsilon) ;

  if (!backMode) {
    if (!givenMomentsMode) {
      error = op.forward(output,moments,data,multipliers,biases) ;
    } else {
      error = op.forwardWithMoment(output,moments,data,multipliers,biases) ;
    }
  } else {
    if (!givenMomentsMode) {
      error = op.backward(derData,
                          derMultipliers,
                          derBiases,
                          moments,
                          data,
                          multipliers,
                          biases,
                          derOutput) ;
    } else {
      error = op.backwardWithMoment(derData,
                                    derMultipliers,
                                    derBiases,
                                    moments,
                                    data,
                                    multipliers,
                                    biases,
                                    derOutput) ;
    }
    return vl::VLE_Success ;
  }
}
#endif

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

