// @file nnbnorm.cu
// @brief Batch normalization block
// @author Sebastien Ehrhardt
// @author Andrea Vedaldi

/*
Copyright (C) 2015-16 Sebastien Ehrhardt and Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "nnbnorm.hpp"
#include "impl/bnorm.hpp"

#if ENABLE_GPU
#include "datacu.hpp"
#endif

#include <assert.h>

using namespace vl ;

/* ---------------------------------------------------------------- */
/*                                                   nnconv_forward */
/* ---------------------------------------------------------------- */

#define DISPATCH(deviceType,type) \
error = vl::impl::bnorm<deviceType,type>::forward \
(context, \
 (type*)output.getMemory(), \
 (type*)moments.getMemory(), \
 (type const*)data.getMemory(), \
 (type*)multipliers.getMemory(), \
 (type*)biases.getMemory(), \
 data.getHeight(), data.getWidth(), data.getDepth(), data.getSize(), \
 epsilon);

#define DISPATCH2(deviceType) \
switch (dataType) { \
case vlTypeFloat : DISPATCH(deviceType, float) ; break ; \
IF_DOUBLE(case vlTypeDouble : DISPATCH(deviceType, double) ; break ;) \
default: assert(false) ; return vlErrorUnknown ; \
}

vl::Error
vl::nnbnorm_forward(vl::Context& context,
                    vl::Tensor output,
                    vl::Tensor moments,
                    vl::Tensor data,
                    vl::Tensor multipliers,
                    vl::Tensor biases,
                    double epsilon)
{
  vl::Error error = vlSuccess ;
  vl::Type dataType = output.getDataType() ;

  switch (output.getDeviceType()) {
    default:
      assert(false) ;
      error = vl::vlErrorUnknown ;
      break ;

    case vl::CPU:
      DISPATCH2(vl::CPU) ;
      break ;

#if ENABLE_GPU
    case vl::GPU:
      DISPATCH2(vl::GPU) ;
      if (error == vlErrorCuda) {
        context.setError(context.getCudaHelper().catchCudaError("GPU")) ;
      }
      break;
#endif
  }
  return context.passError(error, __func__) ;
}

#undef DISPATCH
#define DISPATCH(deviceType,type) \
error = vl::impl::bnorm<deviceType,type>::forward_given_moments \
(context, \
(type*)output.getMemory(), \
(type const*)moments.getMemory(), \
(type const*)data.getMemory(), \
(type*)multipliers.getMemory(), \
(type*)biases.getMemory(), \
data.getHeight(), data.getWidth(), data.getDepth(), data.getSize()) ;

vl::Error
vl::nnbnorm_forward_given_moments(vl::Context& context,
                                  vl::Tensor output,
                                  vl::Tensor moments,
                                  vl::Tensor data,
                                  vl::Tensor multipliers,
                                  vl::Tensor biases)
{
  vl::Error error = vlSuccess ;
  vl::Type dataType = output.getDataType() ;

  switch (output.getDeviceType()) {
    default:
      assert(false) ;
      error = vl::vlErrorUnknown ;
      break ;

    case vl::CPU:
      DISPATCH2(vl::CPU) ;
      break ;

#if ENABLE_GPU
    case vl::GPU:
      DISPATCH2(vl::GPU) ;
      if (error == vlErrorCuda) {
        context.setError(context.getCudaHelper().catchCudaError("nnbnorm_*_forward")) ;
      }
      break;
#endif
  }
  return context.passError(error, "nnbnorm_forward_given_moments") ;
}

/* ---------------------------------------------------------------- */
/*                                                  nnconv_backward */
/* ---------------------------------------------------------------- */

#undef DISPATCH
#define DISPATCH(deviceType,type) \
error = vl::impl::bnorm<deviceType,type>::backward \
(context, \
 (type*)derData.getMemory(), \
 (type*)derMultipliers.getMemory(), \
 (type*)derBiases.getMemory(), \
 (type*)moments.getMemory(), \
 (type const*)data.getMemory(), \
 (type const*)multipliers.getMemory(), \
 (type const*)biases.getMemory(), \
 (type const*)derOutput.getMemory(), \
 data.getHeight(), data.getWidth(), data.getDepth(), data.getSize(), \
 epsilon);

vl::Error
vl::nnbnorm_backward(Context& context,
                     vl::Tensor derData,
                     vl::Tensor derMultipliers,
                     vl::Tensor derBiases,
                     vl::Tensor moments,
                     vl::Tensor data,
                     vl::Tensor multipliers,
                     vl::Tensor biases,
                     vl::Tensor derOutput,
                     double epsilon)
{
  vl::Error error = vl::vlSuccess ;
  vl::Type dataType = derOutput.getDataType() ;

  switch (derOutput.getDeviceType()) {
    default:
      assert(false) ;
      error = vl::vlErrorUnknown ;
      break ;

    case vl::CPU:
      DISPATCH2(vl::CPU) ;
      break ;

#if ENABLE_GPU
    case vl::GPU:
      DISPATCH2(vl::GPU) ;
      if (error == vlErrorCuda) {
        context.setError(context.getCudaHelper().catchCudaError("GPU")) ;
      }
      break;
#endif
  }
  return context.passError(error, __func__) ;
}

#undef DISPATCH
#define DISPATCH(deviceType,type) \
error = vl::impl::bnorm<deviceType,type>::backward_given_moments \
(context, \
(type*)derData.getMemory(), \
(type*)derMultipliers.getMemory(), \
(type*)derBiases.getMemory(), \
(type*)moments.getMemory(), \
(type const*)data.getMemory(), \
(type const*)multipliers.getMemory(), \
(type const*)biases.getMemory(), \
(type const*)derOutput.getMemory(), \
data.getHeight(), data.getWidth(), data.getDepth(), data.getSize(), \
epsilon);

vl::Error
vl::nnbnorm_backward_given_moments(Context& context,
                                   vl::Tensor derData,
                                   vl::Tensor derMultipliers,
                                   vl::Tensor derBiases,
                                   vl::Tensor moments,
                                   vl::Tensor data,
                                   vl::Tensor multipliers,
                                   vl::Tensor biases,
                                   vl::Tensor derOutput,
                                   double epsilon)
{
  vl::Error error = vl::vlSuccess ;
  vl::Type dataType = derOutput.getDataType() ;

  switch (derOutput.getDeviceType()) {
    default:
      assert(false) ;
      error = vl::vlErrorUnknown ;
      break ;

    case vl::CPU:
      DISPATCH2(vl::CPU) ;
      break ;

#if ENABLE_GPU
    case vl::GPU:
      DISPATCH2(vl::GPU) ;
      if (error == vlErrorCuda) {
        context.setError(context.getCudaHelper().catchCudaError("GPU")) ;
      }
      break;
#endif
  }
  return context.passError(error, __func__) ;
}
