// @file nnbilinearsampler.cu
// @brief Bilinear sampler block
// @author Ankush Gupta
// @author Andrea Vedaldi

/*
Copyright (C) 2016- Ankush Gupta and Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "nnbilinearsampler.hpp"
#include "impl/bilinearsampler.hpp"

#if ENABLE_GPU
#include "datacu.hpp"
#endif

#if ENABLE_CUDNN
#include "impl/nnbilinearsampler_cudnn.hpp"
#endif

#include <cstdio>
#include <assert.h>

using namespace vl ;

/* ---------------------------------------------------------------- */
/*                                        nnbilinearsampler_forward */
/* ---------------------------------------------------------------- */

#define DISPATCH(deviceType,type) \
error = vl::impl::bilinearsampler<deviceType,type>::forward \
(context, \
(type*) output.getMemory(), \
(type const*) data.getMemory(), \
(type const*) grid.getMemory(), \
output.getHeight(), output.getWidth(), output.getDepth(), output.getSize(), \
data.getHeight(), data.getWidth(), data.getSize());

#define DISPATCH2(deviceType) \
switch (dataType) { \
case vlTypeFloat : DISPATCH(deviceType, float) ; break ; \
IF_DOUBLE(case vlTypeDouble : DISPATCH(deviceType, double) ; break ;) \
default: assert(false) ; return vlErrorUnknown ; \
}

#define DISPATCHCUDNN(dataType) \
error = vl::impl::nnbilinearsampler_cudnn<dataType>::forward \
( context, output, data, grid ) ;

#define DISPATCHCUDNN2() \
switch (dataType) { \
case vlTypeFloat : DISPATCHCUDNN(vlTypeFloat) ; break ; \
IF_DOUBLE(case vlTypeDouble : DISPATCHCUDNN(vlTypeDouble) ; break ;) \
default: assert(false) ; return vlErrorUnknown ; \
}

vl::Error
vl::nnbilinearsampler_forward(Context& context,
                              Tensor output,
                              Tensor data,
                              Tensor grid)
{
  vl::Error error = vlSuccess ;
  vl::Type dataType = output.getDataType() ;

  switch (output.getDeviceType())
  {
    default:
      assert(false);
      error = vl::vlErrorUnknown ;
      break ;

    case vl::CPU:
      DISPATCH2(vl::CPU) ;
      break ;

#if ENABLE_GPU
    case vl::GPU:
#if ENABLE_CUDNN
    if (context.getCudaHelper().getCudnnEnabled()) {
      DISPATCHCUDNN2() ;
      if (error == vl::vlSuccess) { return error ; }
      if (error != vl::vlErrorUnsupported) { return error ; }
    }
#endif
    DISPATCH2(vl::GPU) ;
    if (error == vlErrorCuda) {
      context.setError(context.getCudaHelper().catchCudaError("GPU")) ;
    }
    break;
#endif
  }
  return context.passError(error, __func__);
}

/* ---------------------------------------------------------------- */
/*                                       nnbilinearsampler_backward */
/* ---------------------------------------------------------------- */

#undef DISPATCH
#define DISPATCH(deviceType,type) \
error = vl::impl::bilinearsampler<deviceType,type>::backward \
(context, \
(type*) derData.getMemory(), \
(type *) derGrid.getMemory(), \
(type const*) data.getMemory(), \
(type const*) grid.getMemory(), \
(type const*) derOutput.getMemory(), \
derOutput.getHeight(), derOutput.getWidth(), derOutput.getDepth(), derOutput.getSize(), \
data.getHeight(), data.getWidth(), data.getSize());

#undef DISPATCHCUDNN
#define DISPATCHCUDNN(dataType) \
error = vl::impl::nnbilinearsampler_cudnn<dataType>::backward \
( context, \
  derData, derGrid, \
  data, grid, \
  derOutput );

vl::Error
vl::nnbilinearsampler_backward(Context& context,
                               Tensor derData,
                               Tensor derGrid,
                               Tensor data,
                               Tensor grid,
                               Tensor derOutput)
{
  vl::Error error = vlSuccess ;
  vl::Device deviceType = derOutput.getDeviceType() ;
  vl::Type dataType = derOutput.getDataType() ;

  switch (derOutput.getDeviceType())
  {
    default:
      assert(false);
      error = vl::vlErrorUnknown ;
      break ;

    case vl::CPU:
      DISPATCH2(vl::CPU) ;
      break ;

#if ENABLE_GPU
    case vl::GPU:
#if ENABLE_CUDNN && (CUDNN_VERSION >= 5000)
    if (context.getCudaHelper().getCudnnEnabled()) {
      DISPATCHCUDNN2() ;
      if (error == vl::vlSuccess) { return error ; }
      if (error != vl::vlErrorUnsupported) { return error ; }
    }
#endif
    DISPATCH2(vl::GPU) ;
    if (error == vlErrorCuda) {
      context.setError(context.getCudaHelper().catchCudaError("GPU")) ;
    }
    break;
#endif
  }
  return context.passError(error, __func__);
}
