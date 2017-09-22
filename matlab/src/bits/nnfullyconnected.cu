// @file nnfullyconnected.cu
// @brief Fully-connected block
// @author Andrea Vedaldi

/*
Copyright (C) 2014-17 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "nnfullyconnected.hpp"
#include "impl/dispatcher.hpp"
#include "impl/blashelper.hpp"
#include "impl/copy.hpp"
#include <cassert>

using namespace vl ;
using namespace vl::nn ;
using namespace vl::impl ;

template<DeviceType deviceType, DataType dataType> struct FullyConnectedForward ;
template<DeviceType deviceType, DataType dataType> struct FullyConnectedBackward ;

// -------------------------------------------------------------------
//                                                             Forward
// -------------------------------------------------------------------

template<DeviceType deviceType, DataType dataType>
struct FullyConnectedForward
{
  vl::ErrorCode operator()
  (FullyConnected &op,
   Tensor &output,
   Tensor const& input,
   Tensor const& filter,
   Tensor const& bias)
  {
    vl::ErrorCode error ;
    typedef typename vl::DataTypeTraits<dataType>::type type ;
    type alpha = 1 ;
    type beta = 0 ;

    if (filter) {
      ptrdiff_t filterVolume = filter.getHeight() * filter.getWidth() * filter.getDepth() ;
      if (input.getSize() == 1) {
        /* one image in the stack */
        error = vl::impl::blas<deviceType,dataType>::gemv
        (op.context,
         't',
         filterVolume, filter.getSize(),
         alpha,
         (type const*)filter.getMemory(), filterVolume,
         (type const*)input.getMemory(), 1,
         beta,
         (type*)output.getMemory(), 1) ;
        if (error != vl::VLE_Success) { goto done ; }
      } else {
        /* multiple images in the stack */
        error = vl::impl::blas<deviceType,dataType>::gemm
        (op.context,
         't', 'n',
         filter.getSize(), input.getSize(), filterVolume,
         alpha,
         (type const*)filter.getMemory(), filterVolume,
         (type const*)input.getMemory(), filterVolume,
         beta,
         (type*)output.getMemory(), filter.getSize()) ;
        if (error != vl::VLE_Success) { goto done ; }
      }
    } else {
      error = vl::impl::operations<deviceType,type>::copy
      ((type*)output.getMemory(),
       (type const*)input.getMemory(),
       input.getNumElements()) ;
    }

    if (bias) {
      type beta = 1 ;
      type const* allOnesMemory = (type*)
      op.context.getAllOnes(deviceType,
                            dataType,
                            input.getSize()) ;
      if (allOnesMemory == NULL) {
        error = op.context.getLastError() ;
        goto done ;
      }
      error = vl::impl::blas<deviceType,dataType>::gemm
      (op.context, 'n', 'n',
       bias.getNumElements(), input.getSize(), 1,
       alpha,
       (type*)bias.getMemory(), bias.getNumElements(),
       allOnesMemory, 1,
       beta,
       (type*)output.getMemory(), bias.getNumElements()) ;
      if (error != vl::VLE_Success) { goto done ; }
    }
  done:
    return op.context.passError(error, __func__) ;
  }
};

// -------------------------------------------------------------------
//                                                           Backward
// -------------------------------------------------------------------

template<DeviceType deviceType, DataType dataType>
struct FullyConnectedBackward
{
  vl::ErrorCode operator()
  (FullyConnected &op,
   vl::Tensor &derInput,
   vl::Tensor &derFilter,
   vl::Tensor &derBias,
   vl::Tensor const &input,
   vl::Tensor const &filter,
   vl::Tensor const &derOutput)
  {
    vl::ErrorCode error ;
    typedef typename vl::DataTypeTraits<dataType>::type type ;
    type alpha = 1 ;
    type beta = 0 ;

    if (filter) {
      ptrdiff_t filterVolume = filter.getHeight() * filter.getWidth() * filter.getDepth() ;

      if (derFilter) {
        error = vl::impl::blas<deviceType,dataType>::gemm
        (op.context,
         'n', 't',
         filterVolume, filter.getSize(), input.getSize(),
         alpha,
         (type*)input.getMemory(), filterVolume,
         (type*)derOutput.getMemory(), filter.getSize(),
         beta,
         (type*)derFilter.getMemory(), filterVolume) ;
        if (error != vl::VLE_Success) { goto done ; }
      }

      if (derInput) {
        error = vl::impl::blas<deviceType,dataType>::gemm
        (op.context,
         'n', 'n',
         filterVolume, input.getSize(), filter.getSize(),
         alpha,
         (type*)filter.getMemory(), filterVolume,
         (type*)derOutput.getMemory(), filter.getSize(),
         beta,
         (type*)derInput.getMemory(), filterVolume) ;
        if (error != vl::VLE_Success) { goto done ; }
      }
    } else {
      vl::impl::operations<deviceType,type>::copy
      ((type*)derInput.getMemory(),
       (type const*)derOutput.getMemory(),
       derOutput.getNumElements()) ;
    }

    if (derBias) {
      auto allOnesMemory = (type const*)
      op.context.getAllOnes(deviceType,
                            dataType,
                            derOutput.getSize()) ;
      if (allOnesMemory == NULL) {
        error = op.context.getLastError() ;
        goto done ;
      }

      error = vl::impl::blas<deviceType, dataType>::gemm
      (op.context,
       'n', 't',
       1, derOutput.getDepth(), derOutput.getSize(),
       alpha,
       (type*)allOnesMemory, 1,
       (type*)derOutput.getMemory(), derOutput.getDepth(),
       beta,
       (type*)derBias.getMemory(), 1) ;
      if (error != vl::VLE_Success) { goto done ; }

    }
  done:
    return op.context.passError(error, __func__) ;
  }
};

// -------------------------------------------------------------------
//                                                              Driver
// -------------------------------------------------------------------

FullyConnected::FullyConnected(Context &context)
: context(context)
{ }

vl::ErrorCode
FullyConnected::forward(Tensor &output,
                        Tensor const& input,
                        Tensor const& filter,
                        Tensor const& bias)
{
  return dispatch<FullyConnectedForward>()
  (*this,output,input,filter,bias) ;
}

vl::ErrorCode
FullyConnected::backward(Tensor &derInput,
                         Tensor &derFilter,
                         Tensor &derBias,
                         Tensor const &input,
                         Tensor const &filter,
                         Tensor const &derOutput)
{
  return dispatch<FullyConnectedBackward>()
  (*this,derInput,derFilter,derBias,input,filter,derOutput) ;
}

