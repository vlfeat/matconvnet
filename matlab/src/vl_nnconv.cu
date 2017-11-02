// @file vl_nnconv.cu
// @brief Convolution block MEX wrapper
// @author Andrea Vedaldi
// @author Max Jaderberg

/*
Copyright (C) 2014 Andrea Vedaldi and Max Jaderberg
Copyright (C) 2015-17 Andrea Vedaldi.

All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "bits/mexutils.h"
#include "bits/datamex.hpp"
#include "bits/nnconv.hpp"

#if ENABLE_GPU
#include "bits/datacu.hpp"
#endif

#include <memory>
#include <cassert>
#include <math.h>

using Int = vl::Int ;

/* option codes */
enum {
  opt_stride = 0,
  opt_padding,
  opt_dilation,
  opt_verbose,
  opt_no_der_data,
  opt_no_der_filters,
  opt_no_der_biases,
  opt_cudnn,
  opt_no_cudnn,
  opt_cudnn_workspace_limit,
  opt_transpose
} ;

/* options */
VLMXOption options [] = {
  {"Stride",                1,   opt_stride                },
  {"Pad",                   1,   opt_padding               },
  {"Padding",               1,   opt_padding               },
  {"Dilate",                1,   opt_dilation              },
  {"Dilation",              1,   opt_dilation              },
  {"Verbose",               0,   opt_verbose               },
  {"NoDerData",             0,   opt_no_der_data           },
  {"NoDerFilters",          0,   opt_no_der_filters        },
  {"NoderBiases",           0,   opt_no_der_biases         },
  {"Cudnn",                 0,   opt_cudnn                 },
  {"NoCudnn",               0,   opt_no_cudnn              },
  {"CudnnWorkSpaceLimit",   1,   opt_cudnn_workspace_limit },
  {0,                       0,   0                         }
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
  IN_DATA = 0, IN_FILTERS, IN_BIASES, IN_DEROUTPUT, IN_END
} ;

enum {
  OUT_RESULT = 0, OUT_DERFILTERS, OUT_DERBIASES, OUT_END
} ;

vl::ErrorCode performConvolution(vl::MexContext& context,
                                 int nout, mxArray *out[],
                                 int nin, mxArray const *in[])
{
  bool backMode = false ;
  bool computeDerData = true ;
  bool computeDerFilters = true ;
  bool computeDerBiases = true ;

  int verbosity = 0 ;
  int opt ;
  int next = IN_END ;
  mxArray const *optarg ;

  vl::nn::Convolution op(context) ;

  if (nin < 3) {
    return context.passError(vl::VLE_IllegalArgument, "There are less than three arguments.") ;
  }

  if (nin > 3 && vlmxIsString(in[3],-1)) {
    next = 3 ;
    backMode = 0 ;
  } else {
    backMode = (nin >= 4) ;
  }

  while ((opt = vlmxNextOption (in, nin, options, &next, &optarg)) >= 0) {
    switch (opt) {
      case opt_verbose : context.setLogLevel(++verbosity) ; break ;
      case opt_stride : MXOPTIVEC(STRIDE,setStride) ; break ;
      case opt_padding : MXOPTIVEC(PADDING,setPadding) ; break ;
      case opt_dilation: MXOPTIVEC(DILATION,setDilation) ; break ;
      case opt_no_der_data : computeDerData = false ; break ;
      case opt_no_der_filters : computeDerFilters = false ; break ;
      case opt_no_der_biases : computeDerBiases = false ; break ;
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
      case opt_cudnn_workspace_limit : {
#if ENABLE_CUDNN
        double x ;
        if (!vlmxIsScalar(optarg) || (x = mxGetScalar(optarg)) < 0) {
          context.passError(vl::VLE_IllegalArgument, "CudnnWorkSpaceLimit is not a non-negative scalar.") ;
        }
        context.getCudaHelper().setCudnnConvolutionFwdPreference
        ((x==mxGetInf() ?
          CUDNN_CONVOLUTION_FWD_PREFER_FASTEST :
          CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT),
         (size_t)x) ;
        context.getCudaHelper().setCudnnConvolutionBwdFilterPreference
        ((x==mxGetInf() ?
          CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST :
          CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT),
         (size_t)x) ;
        context.getCudaHelper().setCudnnConvolutionBwdDataPreference
        ((x==mxGetInf() ?
          CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST :
          CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT),
         (size_t)x) ;
#endif
        break ;
      }
      default: break ;
    }
  }

  vl::MexTensor data(context) ;
  data.init(in[IN_DATA]) ;
  data.reshape(4) ;

  vl::MexTensor filters(context) ;
  filters.init(in[IN_FILTERS]) ;
  filters.reshape(4) ;

  vl::MexTensor biases(context) ;
  biases.init(in[IN_BIASES]) ;

  bool hasFilters = !filters.isEmpty() ;
  bool hasBiases = !biases.isEmpty() ;

  if (!backMode) {
    // Forward mode.
    vl::DeviceType deviceType = data.getDeviceType() ;
    vl::DataType dataType = data.getDataType() ;

    // Compute the size of the output tensor.
    vl::TensorShape outputShape ;
    MXCHECK(op.forwardShape(outputShape,data,filters,biases)) ;

    // Initialize output tensor.
    vl::MexTensor output(context) ;
    output.init(deviceType, dataType, outputShape) ;

    // Perform calculation.
    MXCHECK(op.forward(output,0,data,1,filters,biases)) ;

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
    if (computeDerData) {
      derData.init(deviceType, dataType, data.getShape()) ;
    }

    vl::MexTensor derFilters(context) ;
    if (computeDerFilters && hasFilters) {
      derFilters.init(deviceType, dataType, filters.getShape()) ;
    }

    vl::MexTensor derBiases(context) ;
    if (computeDerBiases && hasBiases) {
      derBiases.init(deviceType, dataType, biases.getShape()) ;
    }

    // Perform calculation.
    MXCHECK(op.backward(derData,derFilters,derBiases,data,filters,derOutput)) ;

    // Return results.
    out[OUT_RESULT] = derData.relinquish() ;
    out[OUT_DERFILTERS] = derFilters.relinquish() ;
    out[OUT_DERBIASES] = derBiases.relinquish() ;
  }
  return vl::VLE_Success ;
}

void mexFunction(int nout, mxArray *out[],
                 int nin, mxArray const *in[])
{
  mexAtExit(atExit) ;
  context.setLogLevel(0) ;
  context.clearLog() ;

  vl::ErrorCode error = performConvolution(context,nout,out,nin,in) ;

  if (context.getLogLevel() > 0) {
    mexPrintf("vl_nnconv:\n") ;
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

