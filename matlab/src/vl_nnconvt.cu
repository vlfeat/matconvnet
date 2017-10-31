// @file vl_nnconvt.cu
// @brief Convolution transpose block MEX wrapper
// @author Andrea Vedaldi

/*
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

using Int = vl::Int ;

/* option codes */
enum {
  opt_upsampling = 0,
  opt_cropping,
  opt_verbose,
  opt_num_groups,
  opt_no_der_data,
  opt_no_der_filters,
  opt_no_der_biases,
  opt_cudnn,
  opt_no_cudnn,
  opt_cudnn_workspace_limit,
} ;

/* options */
VLMXOption  options [] = {
  {"Upsample",              1,   opt_upsampling            },
  {"Upsampling",            1,   opt_upsampling            },
  {"Crop",                  1,   opt_cropping              },
  {"Cropping",              1,   opt_cropping              },
  {"Verbose",               0,   opt_verbose               },
  {"NumGroups",             1,   opt_num_groups            },
  {"NoDerData",             0,   opt_no_der_data           },
  {"NoDerFilters",          0,   opt_no_der_filters        },
  {"NoDerBiases",           0,   opt_no_der_biases         },
  {"CUDNN",                 0,   opt_cudnn                 },
  {"NoCUDNN",               0,   opt_no_cudnn              },
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

#define ERR(code,message) \
context.passError(code,message)

#define CHECK2(x) \
{ vl::ErrorCode err = (x) ; if (err != vl::VLE_Success) { return err ; } }

/* ---------------------------------------------------------------- */
/*                                                       MEX driver */
/* ---------------------------------------------------------------- */

enum {
  IN_DATA = 0, IN_FILTERS, IN_BIASES, IN_DEROUTPUT, IN_END
} ;

enum {
  OUT_RESULT = 0, OUT_DERFILTERS, OUT_DERBIASES, OUT_END
} ;

vl::ErrorCode
performConvolutionTranspose(vl::Context& contetx,
                            int nout, mxArray *out[],
                            int nin, mxArray const *in[])
{
  bool backMode = false ;
  bool computeDerData = true ;
  bool computeDerFilters = true ;
  bool computederBiases = true ;

  int verbosity = 0 ;
  int opt ;
  int next = IN_END ;
  mxArray const *optarg ;

  vl::nn::ConvolutionTranspose op(context) ;

  if (nin < 3) {
    return ERR(vl::VLE_IllegalArgument, "There are less than three arguments.") ;
  }

  if (nin > 3 && vlmxIsString(in[3],-1)) {
    next = 3 ;
    backMode = 0 ;
  } else {
    backMode = (nin >= 4) ;
  }

  while ((opt = vlmxNextOption (in, nin, options, &next, &optarg)) >= 0) {
    switch (opt) {
      case opt_verbose :
        ++ verbosity ;
        context.setLogLevel(verbosity) ;
        break ;

      case opt_upsampling : {
        std::vector<Int> upsampling ;
        if (context.parse(upsampling,optarg) != vl::VLE_Success) {
          return ERR(vl::VLE_IllegalArgument, "Could not set UPSAMPLING:") ;
        }
        CHECK2(op.setUpsampling(upsampling)) ;
        break ;
      }

      case opt_cropping : {
        std::vector<Int> cropping ;
        if (context.parse(cropping,optarg) != vl::VLE_Success) {
          return ERR(vl::VLE_IllegalArgument, "Could not set CROPPING:") ;
        }
        CHECK2(op.setCropping(cropping)) ;
        break ;
      }

      case opt_num_groups : {
        if (!vlmxIsPlainMatrix(optarg,1,1)) {
          return ERR(vl::VLE_IllegalArgument, "NUMGROUPS is not a plain scalar.") ;
        }
        CHECK2(op.setNumFilterGroups((Int)mxGetPr(optarg)[0])) ;
        break;
      }

      case opt_no_der_data : computeDerData = false ; break ;
      case opt_no_der_filters : computeDerFilters = false ; break ;
      case opt_no_der_biases : computederBiases = false ; break ;

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

      case opt_cudnn_workspace_limit :
      {
#if ENABLE_CUDNN
        double x ;
        if (!vlmxIsScalar(optarg) || (x = mxGetScalar(optarg)) < 0) {
          ERR(vl::VLE_IllegalArgument, "CudnnWorkSpaceLimit is not a non-negative scalar.") ;
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
        break ;
#endif
      }
      default: break ;
    }
  }

  vl::MexTensor data(context) ;
  vl::MexTensor filters(context) ;
  vl::MexTensor biases(context) ;

  data.init(in[IN_DATA]) ;
  data.reshape(4) ;

  filters.init(in[IN_FILTERS]) ;
  filters.reshape(4) ;

  biases.init(in[IN_BIASES]) ;

  bool hasFilters = !filters.isEmpty() ;
  bool hasBiases = !biases.isEmpty() ;

  if (!backMode) {
    // Forward mode.
    vl::DeviceType deviceType = data.getDeviceType() ;
    vl::DataType dataType = data.getDataType() ;

    // Compute the size of the output tensor.
    vl::TensorShape outputShape ;
    CHECK2(op.forwardShape(outputShape,data,filters,biases)) ;

    // Initialize output tensor.
    vl::MexTensor output(context) ;
    output.init(deviceType, dataType, outputShape) ;

    // Perform calculation.
    CHECK2(op.forward(output,data,filters,biases)) ;

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
    if (computederBiases && hasBiases) {
      derBiases.init(deviceType, dataType, biases.getShape()) ;
    }

    // Perform calculation.
    CHECK2(op.backward(derData,derFilters,derBiases,data,filters,derOutput)) ;

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

  vl::ErrorCode error = performConvolutionTranspose(context,nout,out,nin,in) ;

  if (context.getLogLevel() > 0) {
    mexPrintf("vl_nnconvt:\n") ;
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

