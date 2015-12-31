// @file nnfullyconnected.cu
// @brief Fully-connected block
// @author Andrea Vedaldi

/*
Copyright (C) 2014-16 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "nnfullyconnected.hpp"
#include "impl/blashelper.hpp"
#include "impl/copy.hpp"
#include <assert.h>

using namespace vl ;

/* ---------------------------------------------------------------- */
/* nnfullyconnected_forward_impl                                    */
/* ---------------------------------------------------------------- */

template<vl::Device deviceType, vl::Type dataType> vl::Error
nnfullyconnected_forward_impl(Context& context,
                              Tensor output,
                              Tensor data,
                              Tensor filters,
                              Tensor biases)
{
  vl::Error error ;
  typedef typename vl::DataTypeTraits<dataType>::type type ;
  type alpha = 1 ;
  type beta = 0 ;

  if (filters) {
    ptrdiff_t filtersVolume = filters.getHeight() * filters.getWidth() * filters.getDepth() ;
    if (data.getSize() == 1) {
      /* one image in the stack */
      error = vl::impl::blas<deviceType,dataType>::gemv
      (context,
       't',
       filtersVolume, filters.getSize(),
       alpha,
       (type const*)filters.getMemory(), filtersVolume,
       (type const*)data.getMemory(), 1,
       beta,
       (type*)output.getMemory(), 1) ;
      if (error != vl::vlSuccess) { goto done ; }
    } else {
      /* multiple images in the stack */
      error = vl::impl::blas<deviceType,dataType>::gemm
      (context,
       't', 'n',
       filters.getSize(), data.getSize(), filtersVolume,
       alpha,
       (type const*)filters.getMemory(), filtersVolume,
       (type const*)data.getMemory(), filtersVolume,
       beta,
       (type*)output.getMemory(), filters.getSize()) ;
      if (error != vl::vlSuccess) { goto done ; }
    }
  } else {
    error = vl::impl::operations<deviceType,type>::copy
    ((type*)output.getMemory(),
     (type const*)data.getMemory(),
     data.getNumElements()) ;
  }

  if (biases) {
    type beta = 1 ;
    type const* allOnesMemory = (type*) context.getAllOnes(deviceType,
                                                           dataType,
                                                           data.getSize()) ;
    if (allOnesMemory == NULL) {
      error = context.getLastError() ;
      goto done ;
    }
    error = vl::impl::blas<deviceType,dataType>::gemm
    (context, 'n', 'n',
     biases.getNumElements(), data.getSize(), 1,
     alpha,
     (type*)biases.getMemory(), biases.getNumElements(),
     allOnesMemory, 1,
     beta,
     (type*)output.getMemory(), biases.getNumElements()) ;
    if (error != vl::vlSuccess) { goto done ; }
  }
done:
  return context.passError(error, __func__) ;
}

/* ---------------------------------------------------------------- */
/* nnfullyconnected_backward_impl                                   */
/* ---------------------------------------------------------------- */

template<vl::Device deviceType, vl::Type dataType> vl::Error
nnfullyconnected_backward_impl(vl::Context& context,
                               vl::Tensor derData,
                               vl::Tensor derFilters,
                               vl::Tensor derBiases,
                               vl::Tensor data,
                               vl::Tensor filters,
                               vl::Tensor derOutput)
{
  vl::Error error ;
  typedef typename vl::DataTypeTraits<dataType>::type type ;
  type alpha = 1 ;
  type beta = 0 ;

  if (filters) {
    ptrdiff_t filtersVolume = filters.getHeight() * filters.getWidth() * filters.getDepth() ;

    if (derFilters) {
      error = vl::impl::blas<deviceType,dataType>::gemm
      (context,
       'n', 't',
       filtersVolume, filters.getSize(), data.getSize(),
       alpha,
       (type*)data.getMemory(), filtersVolume,
       (type*)derOutput.getMemory(), filters.getSize(),
       beta,
       (type*)derFilters.getMemory(), filtersVolume) ;
      if (error != vl::vlSuccess) { goto done ; }
    }

    if (derData) {
      error = vl::impl::blas<deviceType,dataType>::gemm
      (context,
       'n', 'n',
       filtersVolume, data.getSize(), filters.getSize(),
       alpha,
       (type*)filters.getMemory(), filtersVolume,
       (type*)derOutput.getMemory(), filters.getSize(),
       beta,
       (type*)derData.getMemory(), filtersVolume) ;
      if (error != vl::vlSuccess) { goto done ; }
    }
  } else {
    vl::impl::operations<deviceType,type>::copy
    ((type*)derData.getMemory(),
     (type const*)derOutput.getMemory(),
     derOutput.getNumElements()) ;
  }

  if (derBiases) {
    type const* allOnesMemory = (type*) context.getAllOnes(deviceType,
                                                           dataType,
                                                           derOutput.getSize()) ;
    if (allOnesMemory == NULL) {
      error = context.getLastError() ;
      goto done ;
    }

    error = vl::impl::blas<deviceType, dataType>::gemm
    (context,
     'n', 't',
     1, derOutput.getDepth(), derOutput.getSize(),
     alpha,
     (type*)allOnesMemory, 1,
     (type*)derOutput.getMemory(), derOutput.getDepth(),
     beta,
     (type*)derBiases.getMemory(), 1) ;
    if (error != vl::vlSuccess) { goto done ; }

  }
done:
  return context.passError(error, __func__) ;
}

/* ---------------------------------------------------------------- */
/* nnfullyconnected_forward                                         */
/* ---------------------------------------------------------------- */

#define DISPATCH(deviceType, dataType) \
error = nnfullyconnected_forward_impl<deviceType,dataType> \
(context, output, data, filters, biases) ;

#define DISPATCH2(deviceType) \
switch (dataType) { \
case vlTypeFloat : DISPATCH(deviceType, vlTypeFloat) ; break ; \
IF_DOUBLE(case vlTypeDouble : DISPATCH(deviceType, vlTypeDouble) ; break ;) \
default: assert(false) ; return vlErrorUnknown ; \
}

vl::Error
vl::nnfullyconnected_forward(Context& context,
                             Tensor output,
                             Tensor data,
                             Tensor filters,
                             Tensor biases)
{
  vl::Error error = vl::vlSuccess ;
  vl::Type dataType = data.getDataType() ;

  switch (data.getDeviceType()) {
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
      break ;
#endif
  }
  return context.passError(error, __func__) ;
}

/* ---------------------------------------------------------------- */
/* nnfullyconnected_backward                                        */
/* ---------------------------------------------------------------- */

#undef DISPATCH
#define DISPATCH(deviceType, dataType) \
error = nnfullyconnected_backward_impl<deviceType,dataType> \
(context, derData, derFilters, derBiases, data, filters, derOutput) ;

vl::Error
vl::nnfullyconnected_backward(vl::Context& context,
                              vl::Tensor derData,
                              vl::Tensor derFilters,
                              vl::Tensor derBiases,
                              vl::Tensor data,
                              vl::Tensor filters,
                              vl::Tensor derOutput)
{
  vl::Error error = vl::vlSuccess ;
  vl::Type dataType = data.getDataType() ;

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
      break ;
#endif
  }
  return context.passError(error, __func__) ;
}


