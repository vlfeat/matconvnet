// @file nnbias.cu
// @brief Bias block
// @author Andrea Vedaldi

/*
Copyright (C) 2015-16 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "nnbias.hpp"
#include "impl/nnbias_blas.hpp"
#if ENABLE_CUDNN
#include "impl/nnbias_cudnn.hpp"
#endif
#include <assert.h>

using namespace vl ;

/* ---------------------------------------------------------------- */
/* Forward                                                          */
/* ---------------------------------------------------------------- */

#define DISPATCH(deviceType,dataType) \
status = vl::impl::nnbias_forward_blas<deviceType,dataType> \
(context, output, outputMult, data, dataMult, biases, biasesMult) ;

#define DISPATCH2(deviceType) \
switch (dataType) { \
case vlTypeFloat : DISPATCH(deviceType,vlTypeFloat) ; break ; \
IF_DOUBLE(case vlTypeDouble : DISPATCH(deviceType,vlTypeDouble) ; break ;) \
default: assert(false) ; return vlErrorUnknown ; \
}

#define DISPATCHCUDNN(dataType) \
status = vl::impl::nnbias_cudnn<dataType>::forward \
(context, output, outputMult, data, dataMult, biases, biasesMult) ;

#define DISPATCHCUDNN2() \
switch (dataType) { \
case vlTypeFloat : DISPATCHCUDNN(vlTypeFloat) ; break ; \
IF_DOUBLE(case vlTypeDouble : DISPATCHCUDNN(vlTypeDouble) ; break ;) \
default: assert(false) ; return vlErrorUnknown ; \
}

vl::Error
vl::nnbias_forward(vl::Context& context,
                   vl::Tensor output, double outputMult,
                   vl::Tensor data, double dataMult,
                   vl::Tensor biases, double biasesMult)
{
  vl::Error status = vlSuccess ;
  vl::Type dataType = output.getDataType() ;

  switch (output.getDeviceType()) {
    default:
      assert(false) ;
      status = vl::vlErrorUnknown ;
      break ;

    case vl::CPU:
      DISPATCH2(vl::CPU) ;
      break ;

#if ENABLE_GPU
    case vl::GPU:
#if ENABLE_CUDNN
      if (context.getCudaHelper().getCudnnEnabled()) {
        DISPATCHCUDNN2() ;
        if (status == vl::vlSuccess) { return status ; }
        if (status != vl::vlErrorUnsupported) { goto done ; }
        /* this case was not supported by CUDNN -- fallback */
      }
#endif
      DISPATCH2(vl::GPU) ;
      break ;
#endif
  }
#if ENABLE_CUDNN
done:
#endif
  return context.passError(status, __func__) ;
}

/* ---------------------------------------------------------------- */
/* Backward                                                         */
/* ---------------------------------------------------------------- */

#undef DISPATCH
#define DISPATCH(deviceType,dataType) \
status = vl::impl::nnbias_backward_blas<deviceType,dataType> \
(context, derData, derDataMult, derBiases, derBiasesMult, derOutput, derOutputMult) ;

#undef DISPATCHCUDNN
#define DISPATCHCUDNN(dataType) \
status = vl::impl::nnbias_cudnn<dataType>::backward \
(context, derData, derDataMult, derBiases, derBiasesMult, derOutput, derOutputMult) ;

vl::Error
vl::nnbias_backward(vl::Context& context,
                    vl::Tensor derData, double derDataMult,
                    vl::Tensor derBiases, double derBiasesMult,
                    vl::Tensor derOutput, double derOutputMult)
{
  vl::Error status = vlSuccess ;
  vl::Type dataType = derOutput.getDataType() ;

  switch (derOutput.getDeviceType()) {
    default:
      assert(false) ;
      status = vl::vlErrorUnknown ;
      break ;

    case vl::CPU:
      DISPATCH2(vl::CPU) ;
      break ;

#if ENABLE_GPU
    case vl::GPU:
#if ENABLE_CUDNN
      if (context.getCudaHelper().getCudnnEnabled()) {
        DISPATCHCUDNN2() ;
        if (status == vl::vlSuccess) { return status ; }
        if (status != vl::vlErrorUnsupported) { goto done ; }
        /* this case was not supported by CUDNN -- fallback */
      }
#endif
      DISPATCH2(vl::GPU) ;
      break ;
#endif
  }
#if ENABLE_CUDNN
done:
#endif
  return context.passError(status, __func__) ;
}

