// @file nnpooling.cu
// @brief Pooling block
// @author Andrea Vedaldi

/*
Copyright (C) 2014-16 Andrea Vedaldi and Karel Lenc.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "nnpooling.hpp"
#include "impl/pooling.hpp"

#if ENABLE_GPU
#include "datacu.hpp"
#endif

#if ENABLE_CUDNN
#include "impl/nnpooling_cudnn.hpp"
#endif
#include <assert.h>

using namespace vl ;

/* ---------------------------------------------------------------- */
/*                                                nnpooling_forward */
/* ---------------------------------------------------------------- */

#define DISPATCH(deviceType, op, type) \
status = vl::impl::op<deviceType, type>::forward \
((type*)output.getMemory(), (type const*)data.getMemory(), \
data.getHeight(), data.getWidth(), data.getDepth() * data.getSize(), \
poolHeight, poolWidth, \
strideY, strideX, \
padTop, padBottom, \
padLeft, padRight) ;

#define DISPATCH2(deviceType, op) \
switch (dataType) { \
case vlTypeFloat : DISPATCH(deviceType, op, float) ; break ; \
IF_DOUBLE(case vlTypeDouble : DISPATCH(deviceType, op, double) ; break ;) \
default: assert(false) ; return vlErrorUnknown ; \
}

#define DISPATCH3(deviceType) \
switch (method) { \
case vlPoolingAverage : DISPATCH2(deviceType, pooling_average) ; break ; \
case vlPoolingMax : DISPATCH2(deviceType, pooling_max) ; break ; \
default: assert(false) ; return vlErrorUnknown ; \
}

#define DISPATCHCUDNN(dataType) \
status = vl::impl::nnpooling_cudnn<dataType>::forward \
(context, output, data, \
method, \
poolHeight, poolWidth, \
strideY, strideX, \
padTop, padBottom, \
padLeft, padRight) ;

#define DISPATCHCUDNN2() \
switch (dataType) { \
case vlTypeFloat : DISPATCHCUDNN(vlTypeFloat) ; break ; \
IF_DOUBLE(case vlTypeDouble : DISPATCHCUDNN(vlTypeDouble) ; break ;) \
default: assert(false) ; return vlErrorUnknown ; \
}

vl::Error
vl::nnpooling_forward(vl::Context& context,
                      vl::Tensor output,
                      vl::Tensor data,
                      PoolingMethod method,
                      int poolHeight, int poolWidth,
                      int strideY, int strideX,
                      int padTop, int padBottom,
                      int padLeft, int padRight)
{
  vl::Error status = vlSuccess ;
  vl::Device deviceType = output.getDeviceType() ;
  vl::Type dataType = output.getDataType() ;

  switch (deviceType) {
    default:
      assert(false) ;
      return vl::vlErrorUnknown ;

    case vl::CPU:
      DISPATCH3(vl::CPU) ;
      break ;

#ifdef ENABLE_GPU
    case vl::GPU:
#if ENABLE_CUDNN
      if (context.getCudaHelper().getCudnnEnabled()) {
        DISPATCHCUDNN2() ;
        if (status == vl::vlSuccess) { return status ; }
        if (status != vl::vlErrorUnsupported) { return status ; }
        /* this case was not supported by CUDNN -- fallback */
      }
#endif
      DISPATCH3(GPU) ;
      if (status == vlErrorCuda) {
        context.setError(context.getCudaHelper().catchCudaError(__func__)) ;
      }
      break ;
#endif
  }
  return context.passError(status, "nnpooling_forward") ;
}

/* ---------------------------------------------------------------- */
/*                                               nnpooling_backward */
/* ---------------------------------------------------------------- */

#undef DISPATCH
#undef DISPATCH2

// backward max and average want slightly differet argument lists

#define DISPATCH_pooling_average(deviceType, type) \
status = vl::impl::pooling_average<deviceType, type>::backward \
((type*)derData.getMemory(), (type const*)derOutput.getMemory(), \
derData.getHeight(), derData.getWidth(), derData.getDepth() * derData.getSize(), \
poolHeight, poolWidth, \
strideY, strideX, \
padTop, padBottom, \
padLeft, padRight) ;

#define DISPATCH_pooling_max(deviceType, type) \
status = vl::impl::pooling_max<deviceType, type>::backward \
((type*)derData.getMemory(), (type const*)data.getMemory(), (type const*)derOutput.getMemory(), \
derData.getHeight(), derData.getWidth(), derData.getDepth() * derData.getSize(), \
poolHeight, poolWidth, \
strideY, strideX, \
padTop, padBottom, \
padLeft, padRight) ;

#define DISPATCH2(deviceType, op) \
switch (dataType) { \
case vlTypeFloat : DISPATCH_ ## op (deviceType, float) ; break ; \
IF_DOUBLE(case vlTypeDouble : DISPATCH_ ## op (deviceType, double) ; break ;) \
default: assert(false) ; return vlErrorUnknown ; \
}

vl::Error
vl::nnpooling_backward(Context& context,
                       Tensor derData,
                       Tensor data,
                       Tensor derOutput,
                       PoolingMethod method,
                       int poolHeight, int poolWidth,
                       int strideY, int strideX,
                       int padTop, int padBottom,
                       int padLeft, int padRight)
{
  vl::Error status = vlSuccess ;
  vl::Device deviceType = derOutput.getDeviceType() ;
  vl::Type dataType = derOutput.getDataType() ;

  switch (deviceType) {
    default:
      assert(false) ;
      return vl::vlErrorUnknown ;

    case vl::CPU:
      DISPATCH3(vl::CPU) ;
      break ;

#if ENABLE_GPU
    case vl::GPU:
#if ENABLE_CUDNN
      if (context.getCudaHelper().getCudnnEnabled()) {
        /*
         Unfortunately CuDNN requires both the input and the output pooling arrays
         to be available for computing derivatives, whereas MatConvNet only requires the input one.
         */
      }
#endif
      DISPATCH3(vl::GPU) ;
      if (status == vlErrorCuda) {
        context.setError(context.getCudaHelper().catchCudaError("pooling_*::backward")) ;
      }
      break ;
#endif
  }
  return context.passError(status, "nnpooling_backward") ;
}
