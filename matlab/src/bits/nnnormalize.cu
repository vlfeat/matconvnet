// @file nnnormalize.cu
// @brief Normalization block
// @author Andrea Vedaldi

/*
Copyright (C) 2014-16 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "nnnormalize.hpp"
#include "impl/normalize.hpp"

#if ENABLE_GPU
#include "datacu.hpp"
#endif

#if ENABLE_CUDNN
//#include "impl/normalize_cudnn.hpp"
#endif
#include <assert.h>

using namespace vl ;

/* ---------------------------------------------------------------- */
/*                                                    nnlrn_forward */
/* ---------------------------------------------------------------- */

#define DISPATCH(deviceType, type) \
error = vl::impl::lrn<deviceType,type>::forward \
((type*)output.getMemory(), (type const*)data.getMemory(), \
data.getHeight(), data.getWidth(), data.getDepth(), data.getSize(), \
normDetph, kappa, alpha, beta) ;

#define DISPATCH2(deviceType) \
switch (dataType) { \
case vlTypeFloat : DISPATCH(deviceType, float) ; break ; \
IF_DOUBLE(case vlTypeDouble : DISPATCH(deviceType, double) ; break ;) \
default: assert(false) ; return vlErrorUnknown ; \
}

vl::Error
vl::nnlrn_forward(vl::Context& context,
                  vl::Tensor output,
                  vl::Tensor data,
                  size_t normDetph,
                  double kappa, double alpha, double beta)
{
  vl::Error error = vl::vlSuccess ;
  vl::Type dataType = output.getDataType() ;

  switch (output.getDeviceType()) {
    default:
      assert(false) ;
      return vl::vlErrorUnknown ;

    case vl::CPU:
      DISPATCH2(vl::CPU) ;
      break ;

#ifdef ENABLE_GPU
    case vl::GPU:
#if ENABLE_CUDNN
      if (context.getCudaHelper().getCudnnEnabled()) {
        /*
         error = vl::impl::nnlrn_forward_cudnn<float>(context, output, data, filters, biases,
         strideY, strideX,
         padTop, padBottom,
         padLeft, padRight) ;
         if (error == vl::vlSuccess) { return error ; }
         if (error != vl::UNSUPPORTED) { return error ; }
         */
        /* this case was not supported by CUDNN -- fallback */
      }
#endif
      DISPATCH2(vl::GPU) ;
      if (error != vl::vlSuccess) { context.getCudaHelper().catchCudaError(__func__) ; }
      break ;
#endif
  }
  if (error != vl::vlSuccess) {
    context.setError(error, __func__) ;
  }
  return error ;
}

/* ---------------------------------------------------------------- */
/*                                                   nnlrn_backward */
/* ---------------------------------------------------------------- */

#undef DISPATCH

#define DISPATCH(deviceType, type) \
error = vl::impl::lrn<deviceType,type>::backward \
((type*)derData.getMemory(), (type const*)data.getMemory(), (type const*)derOutput.getMemory(), \
data.getHeight(), data.getWidth(), data.getDepth(), data.getSize(), \
normDetph, kappa, alpha, beta) ;

vl::Error
vl::nnlrn_backward(vl::Context& context,
                   vl::Tensor derData,
                   vl::Tensor data,
                   vl::Tensor derOutput,
                   size_t normDetph,
                   double kappa, double alpha, double beta)
{
  vl::Error error = vl::vlSuccess ;
  vl::Type dataType = derOutput.getDataType() ;

  switch (derOutput.getDeviceType()) {
    default:
      assert(false) ;
      return vl::vlErrorUnknown ;

    case vl::CPU:
      DISPATCH2(vl::CPU) ;
      break ;

#if ENABLE_GPU
    case vl::GPU:
#if ENABLE_CUDNN
      if (context.getCudaHelper().getCudnnEnabled()) {
        /*
         error = vl::impl::nnlrn_backward_cudnn<float>(context, output, data, filters, biases,
         strideY, strideX,
         padTop, padBottom,
         padLeft, padRight) ;
         if (error == vl::vlSuccess) { return error ; }
         if (error != vl::UNSUPPORTED) { return error ; }
         */
        /* this case was not supported by CUDNN -- fallback */
      }
#endif
      DISPATCH2(vl::GPU) ;
      if (error != vl::vlSuccess) { context.getCudaHelper().catchCudaError(__func__) ; }
      break ;
#endif
  }
  if (error != vl::vlSuccess) {
    context.setError(error, __func__) ;
  }
  return error ;
}
